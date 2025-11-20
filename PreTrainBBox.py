import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import copy
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from codecarbon import track_emissions


# Import datasets and models
from data import BBoxTeacherStudentDataset_SUN
from loss import DINOLoss, AttentionOverlapLoss
from model_vit import ClassificationModel, load_pretrained_and_apply_lora

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# histology_to_int_dict = {
#     '0-IIa': 0,
#     '0-Ip': 1,
#     '0-Is': 1,
#     #'Negative': 3,
# }

# int_to_histology_dict = {
#     0: '0-IIa',
#     1: '0-Ip',
#     2: '0-Is'}



# int_to_histology_dict = {
#     0: '0-IIa',
#     1: '0-Ip',
#     1: '0-Is'}

# int_to_histology_dict = {
#     0: '0-II',
#     1: '0-I',
# }


@track_emissions(offline=True, country_iso_code="ITA", project_name="training")
def pretrain_bbox_classifier_overlap(
    args,
    student_model: nn.Module,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    overlap_loss_fn
    ):
        """
        Pretraining loop using only Attention Overlap Loss, with validation,
        LR scheduler on val loss, early stopping (args.patience), and best-model saving.
        """
        student_model.train()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(args.pretrain_epochs):
            # ——— TRAIN ———
            train_loss = 0.0
            for _, student_img, *rest in tqdm(train_dataloader, desc=f'Pretrain Epoch {epoch+1}/{args.pretrain_epochs}'):
                optimizer.zero_grad()
                student_img = student_img.to(args.device)

                # get attention map
                attn_map = (
                    student_model.module.get_cls_attention_map(student_img)
                    if isinstance(student_model, nn.DataParallel)
                    else student_model.get_cls_attention_map(student_img)
                )
                attn_map = F.interpolate(
                    attn_map.unsqueeze(1),
                    size=(224,224),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(1)

                bboxes = torch.stack(rest[-4:], dim=1).to(attn_map.device)  # assumes x1,y1,x2,y2 are last
                loss = overlap_loss_fn(attn_map, bboxes)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                torch.cuda.empty_cache()

            avg_train_loss = train_loss / len(train_dataloader)
            print(f"[Epoch {epoch+1}] Train OverlapLoss: {avg_train_loss:.4f}")

            # ——— VALIDATION ———
            student_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, student_img, *rest in val_dataloader:
                    student_img = student_img.to(args.device)
                    attn_map = (
                        student_model.module.get_cls_attention_map(student_img)
                        if isinstance(student_model, nn.DataParallel)
                        else student_model.get_cls_attention_map(student_img)
                    )
                    attn_map = F.interpolate(
                        attn_map.unsqueeze(1),
                        size=(224,224),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(1)
                    bboxes = torch.stack(rest[-4:], dim=1).to(attn_map.device)
                    val_loss += overlap_loss_fn(attn_map, bboxes).item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"[Epoch {epoch+1}] Val   OverlapLoss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                save_path = os.path.join(os.getcwd(), args.store_name, 'best_model_pretrain_overlap_only.pth')
                state = (student_model.module if isinstance(student_model, nn.DataParallel) else student_model).state_dict()
                torch.save(state, save_path)
                print(f"→ Saved new best Overlap model (val={best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter}/{args.patience} epochs.")

                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break


    

def test_model(args, model, test_dataloader, ce_loss_fn, sun=False, num_classes=3):
    """
    Evaluate the model on the test dataset and store metrics.
    Computes average loss, overall accuracy, per-class accuracy,
    confusion matrix and classification report; then saves these to files.
    """
    model.eval()
    all_preds = []
    all_labels = []
    image_names = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for teacher_image, student_image, histology_label, histology, image_path, _, _, _, _ in tqdm(test_dataloader, desc="Testing"):
            # Ignore teacher_image
            student_image = student_image.to(args.device)
            labels = torch.argmax(histology_label, dim=1).to(args.device)
            logits = model(student_image)
            logits = logits.to(args.device)

            
            loss = ce_loss_fn(logits, labels)

            batch_size = student_image.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            image_names.extend(image_path)

    avg_loss = total_loss / total_samples
    overall_accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / total_samples

    #num_classes = int(max(max(all_labels), max(all_preds)) + 1)
    per_class_acc = {}
    for i in range(num_classes):
        indices = np.where(np.array(all_labels) == i)[0]
        if len(indices) > 0:
            class_acc = 100.0 * np.sum(np.array(all_preds)[indices] == i) / len(indices)
            per_class_acc[i] = class_acc
        else:
            per_class_acc[i] = None

    cm = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, output_dict=True)

    # Inverse-frequency class-weighted accuracy
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    # Count samples and correct predictions per class
    class_total = {i: 0 for i in range(num_classes)}
    class_correct = {i: 0 for i in range(num_classes)}

    for i in range(len(all_labels_np)):
        label = all_labels_np[i]
        pred = all_preds_np[i]
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    # Compute inverse-frequency weights
    class_weights = torch.tensor(
        [1.0 / class_total[c] if class_total[c] > 0 else 0.0 for c in range(num_classes)],
        dtype=torch.float32
    )
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

    # Compute weighted accuracy
    weighted_correct = sum(class_correct[c] * class_weights[i].item() for i, c in enumerate(class_total))
    weighted_total = sum(class_total[c] * class_weights[i].item() for i, c in enumerate(class_total))
    weighted_accuracy = 100.0 * (weighted_correct / weighted_total if weighted_total > 0 else 0.0)


    # Macro accuracy (equal weight to each class)
    valid_class_accs = [acc for acc in per_class_acc.values() if acc is not None]
    macro_accuracy = np.mean(valid_class_accs)

    # Balanced accuracy from sklearn (based on recall)
    balanced_accuracy = 100.0 * balanced_accuracy_score(all_labels_np, all_preds_np)


    results = {
        "average_loss": avg_loss,
        "overall_accuracy": overall_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "macro_accuracy": macro_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class_accuracy": per_class_acc,
        "classification_report": cls_report
    }

    output_dir = os.path.join(os.getcwd(), args.store_name)
    if sun:
        output_dir = os.path.join(output_dir, "SUN")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    df = pd.DataFrame({
        "image_name": image_names,
        "true_label": all_labels,
        "predicted_label": all_preds
    })
    csv_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)

    print(f"Test metrics saved to {json_path}")
    print(f"Confusion matrix saved to {cm_path}")
    print(f"Predictions saved to {csv_path}")

    return results, cm, df

def evaluate_model(args, model, dataloader, ce_loss_fn):
    """
    Evaluate the model on a given dataloader.
    Returns average CrossEntropy loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for teacher_image, student_image, histology_label, histology, image_path, _, _, _, _ in tqdm(dataloader, desc='Evaluation'):
            # Ignore teacher_image
            student_image = student_image.to(args.device)
            labels = histology_label.to(args.device)
            logits = model(student_image)
            logits = logits.to(args.device)
            loss = ce_loss_fn(logits, labels)

            batch_size = student_image.size(0)
            total_loss += loss.item() * batch_size
            labels = torch.argmax(labels, dim=1)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy

@track_emissions(offline=True, country_iso_code="ITA", project_name="training")
def train_bbox_classifier(args, model, train_dataloader, val_dataloader, optimizer, scheduler, ce_loss_fn):
    """
    Training loop for classification using the BBox Teacher-Student Dataset.
    Uses only CrossEntropy loss.
    """
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    patience = getattr(args, 'patience', 20)

    print(f'Number of epochs: {args.num_epochs}')
    wandb.init(project='dinov2_pretraining', name=args.store_name)

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        print(f'Epoch: {epoch}')
        # if first epoch set lr to 1e-4
        
        


        for i, (teacher_image, student_image, histology_label, histology, image_path, x1, y1, x2, y2) in enumerate(
                tqdm(train_dataloader, desc=f'Epoch {epoch}/{args.num_epochs}')):
            optimizer.zero_grad()
            # Ignore teacher_image
            student_image = student_image.to(args.device)
            labels = histology_label.to(args.device)

            logits = model(student_image)
            
            logits = logits.to(args.device)
            loss = ce_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            labels = torch.argmax(labels, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += student_image.size(0)
            
            torch.cuda.empty_cache()

        # if epoch == 0 reset the lr to the one in the arguments


        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_train_acc = 100.0 * epoch_correct / epoch_samples

        print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_acc:.2f}%")
        wandb.log({'Train Loss': avg_train_loss, 'Train Accuracy': avg_train_acc, 'Epoch': epoch})

        val_loss, val_acc = evaluate_model(args, model, val_dataloader, ce_loss_fn)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        wandb.log({'Val Loss': val_loss, 'Val Accuracy': val_acc, 'Epoch': epoch})
        scheduler.step(val_loss)

        # Save the best model based on validation loss.
        if val_loss < best_val_loss:# and epoch > 5:  # Save only after 5 epochs
            best_val_loss = val_loss
            save_path = os.path.join(os.getcwd(), args.store_name, 'best_model.pth')
            # If using DataParallel, save the underlying module.
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            with open(os.path.join(os.getcwd(), args.store_name, 'best_loss.txt'), 'w') as f:
                f.write(f'Best Validation Loss: {best_val_loss:.4f}')
                f.write(f'Best Epoch: {epoch}\n')
            print(f"Saved best model at Epoch {epoch} with Validation Loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break
            
        # Save the final model at each epoch.
        final_path = os.path.join(os.getcwd(), args.store_name, 'final_model.pth')
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, final_path)





def save_attention_masks_and_images(
    args,
    model: nn.Module,
    dataloader,
    output_subdir="attention_visuals",
):
    """
    For each batch:
     - compute both cls-attention and rollout-attention
     - normalize and upsample each to 224×224
     - overlay each on the input image and save (max 10 per case per map)
     - record overlap stats and at the end print & save CSV
    """
    model.eval()
    save_dir = os.path.join(os.getcwd(), args.store_name, output_subdir)
    os.makedirs(save_dir, exist_ok=True)

    # inverse imagenet-normalization
    inv_normalize = T.Normalize(
        mean=[-m/s for m, s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
        std=[1/s for s in [0.229,0.224,0.225]]
    )

    results = []
    # counters per case & per map
    counters = {
        "cls":    {"CI":0,"CO":0,"WI":0,"WO":0},
        "rollout":{"CI":0,"CO":0,"WI":0,"WO":0},
    }
    max_per_group = 10

    with torch.no_grad():
        for idx, (
            teacher_image, student_image,
            histology_label, histology,
            image_path, x1, y1, x2, y2
        ) in enumerate(tqdm(dataloader, desc="Saving masks")):
            student_image = student_image.to(args.device)
            true_labels = histology_label.argmax(1).to(args.device)
            logits = model(student_image)
            preds = logits.argmax(1)

            # compute both attention maps
            def get_map(fn_name):
                if isinstance(model, nn.DataParallel):
                    fn = getattr(model.module, fn_name)
                else:
                    fn = getattr(model, fn_name)
                m = fn(student_image)
                # upsample & normalize per-sample
                m = F.interpolate(
                    m.unsqueeze(1), size=(224,224),
                    mode="bilinear", align_corners=False
                ).squeeze(1)
                flat = m.view(m.size(0),-1)
                mn = flat.min(dim=1,keepdim=True)[0]
                mx = flat.max(dim=1,keepdim=True)[0]
                norm = (flat - mn) / (mx - mn + 1e-8)
                return norm.view(-1,224,224)

            cls_maps    = get_map("get_cls_attention_map")
            rollout_maps= get_map("get_attention_rollout")

            for b_idx, (
                img_tensor, cls_mask, roll_mask,
                label, pred, path, x1i, y1i, x2i, y2i
            ) in enumerate(zip(
                student_image.cpu(), cls_maps.cpu(), rollout_maps.cpu(),
                true_labels.cpu(), preds.cpu(), image_path, x1, y1, x2, y2
            )):
                base = os.path.splitext(os.path.basename(path))[0]
                img = inv_normalize(img_tensor)

                # find max-position
                def maxpos(mask):
                    flat_idx = mask.argmax().item()
                    return divmod(flat_idx,224)
                yC,xC = maxpos(cls_mask)
                yR,xR = maxpos(roll_mask)

                insideC = (x1i <= xC <= x2i) and (y1i <= yC <= y2i)
                insideR = (x1i <= xR <= x2i) and (y1i <= yR <= y2i)
                correct = (pred==label).item()

                # compute real overlap proportion
                def overlap(mask, inside):
                    # normalize sum=1
                    mn = mask.sum()+1e-8
                    #sub = mask[y1i:y2i+1, x1i:x2i+1].sum().item()
                    sub = mask[int(y1i):int(y2i)+1, int(x1i):int(x2i)+1].sum().item()
                    return sub/mn

                overlapC = overlap(cls_mask, insideC)
                overlapR = overlap(roll_mask, insideR)

                # store results
                results.append({
                    "image": base,
                    "true_label":label.item(),
                    "predicted_label":pred.item(),
                    "correct":bool(correct),
                    "inside_cls":bool(insideC),
                    "inside_rollout":bool(insideR),
                    "overlap_cls":overlapC,
                    "overlap_rollout":overlapR,
                })

                # case code: e.g. "CI" = correct & inside
                def case_code(corr, ins):
                    if corr and ins:   return "CI"
                    if corr and not ins:return "CO"
                    if not corr and ins:return "WI"
                    return "WO"

                for map_type, mask, inside_flag in [
                    ("cls", cls_mask, insideC),
                    ("rollout", roll_mask, insideR)
                ]:
                    case = case_code(correct, inside_flag)
                    if counters[map_type][case] < max_per_group:
                        counters[map_type][case] += 1
                        # save raw image
                        img_path = os.path.join(save_dir, f"{base}_{map_type}_image_{case}.png")
                        save_image(img, img_path)
                        # overlay
                        plt.figure(figsize=(224/100,224/100))
                        plt.imshow(img.permute(1,2,0).numpy())
                        plt.imshow(mask.numpy(), cmap="jet", alpha=0.3)
                        plt.title(f"GT:{label.item()} Pred:{pred.item()}")
                        plt.axis("off")
                        ove_path = os.path.join(save_dir, f"{base}_{map_type}_overlay_{case}.png")
                        plt.savefig(ove_path, bbox_inches="tight", pad_inches=0)
                        plt.close()

    # --- After loop: summary & CSV ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "attention_correlation.csv")
    df.to_csv(csv_path, index=False)

    # print overall stats
    total = len(df)
    bal_acc = balanced_accuracy_score(df["true_label"], df["predicted_label"])
    print(f"Total samples: {total}, Overall balanced-acc: {bal_acc:.2%}")

    for m in ["cls","rollout"]:
        inside = df[f"inside_{m}"]
        if inside.any():
            acc_in = (df[inside]["true_label"] == df[inside]["predicted_label"]).mean()
            print(f"{m.upper()} INSIDE bbox acc: {acc_in:.2%}")
        if (~inside).any():
            acc_out = (df[~inside]["true_label"] == df[~inside]["predicted_label"]).mean()
            print(f"{m.upper()} OUTSIDE bbox acc: {acc_out:.2%}")

    print(f"Saved visuals & CSV to: {save_dir}")

def save_attention_masks_and_images_NEW(
    args,
    model: nn.Module,
    dataloader,
    output_subdir="cls_attention_visuals_NEW",
    overlap_threshold=0.5,  # << threshold can be tuned
):
    """
    Compute CLS attention maps, upsample to 224×224, compute overlap with GT bbox,
    and save attention overlays for max 10 examples per case (CI, CO, WI, WO),
    where 'inside' is defined by overlap > threshold.
    """


    model.eval()
    save_dir = os.path.join(os.getcwd(), args.store_name, output_subdir)
    #delete old directory if exists
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Inverse ImageNet normalization
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )

    results = []
    counters = {"CI": 0, "CO": 0, "WI": 0, "WO": 0}
    max_per_group = 10

    with torch.no_grad():
        for (
            teacher_image, student_image,
            histology_label, histology,
            image_path, x1, y1, x2, y2
        ) in tqdm(dataloader, desc="Saving CLS attention masks"):
            student_image = student_image.to(args.device)
            true_labels = histology_label.argmax(1).to(args.device)
            logits = model(student_image)
            preds = logits.argmax(1)

            # get_cls_attention_map
            if isinstance(model, nn.DataParallel):
                attn_map = model.module.get_cls_attention_map(student_image)
            else:
                attn_map = model.get_cls_attention_map(student_image)

            # Interpolate and normalize
            attn_map = F.interpolate(
                attn_map.unsqueeze(1), size=(224, 224),
                mode="bilinear", align_corners=False
            ).squeeze(1)

            flat = attn_map.view(attn_map.size(0), -1)
            min_val = flat.min(dim=1, keepdim=True)[0]
            max_val = flat.max(dim=1, keepdim=True)[0]
            attn_map = (flat - min_val) / (max_val - min_val + 1e-8)
            attn_map = attn_map.view(-1, 224, 224)

            for img_tensor, cls_mask, label, pred, path, x1i, y1i, x2i, y2i in zip(
                student_image.cpu(), attn_map.cpu(),
                true_labels.cpu(), preds.cpu(),
                image_path, x1, y1, x2, y2
            ):
                base = os.path.splitext(os.path.basename(path))[0]
                img = inv_normalize(img_tensor)

                correct = (pred == label).item()

                # Compute normalized attention map and overlap
                norm_mask = cls_mask / (cls_mask.sum() + 1e-8)
                x1i, y1i, x2i, y2i = map(int, [x1i, y1i, x2i, y2i])
                x1i, x2i = max(0, x1i), min(223, x2i)
                y1i, y2i = max(0, y1i), min(223, y2i)

                overlap = norm_mask[y1i:y2i + 1, x1i:x2i + 1].sum().item()
                inside = overlap > overlap_threshold

                results.append({
                    "image": base,
                    "true_label": label.item(),
                    "predicted_label": pred.item(),
                    "correct": correct,
                    "inside_cls": inside,
                    "overlap_cls": overlap,
                })

                # Assign case
                def case_code(c, i):
                    if c and i: return "CI"
                    if c and not i: return "CO"
                    if not c and i: return "WI"
                    return "WO"

                case = case_code(correct, inside)
                if counters[case] < max_per_group:
                    counters[case] += 1
                    # save image and overlay
                    save_image(img, os.path.join(save_dir, f"{base}_cls_image_{case}.png"))
                    plt.figure(figsize=(224 / 100, 224 / 100))
                    plt.imshow(img.permute(1, 2, 0).numpy())
                    plt.imshow(cls_mask.numpy(), cmap="jet", alpha=0.3)
                    # use mapping to get the correct label
                    plt.title(f"GT: {int_to_histology_dict[label.item()]} Pred: {int_to_histology_dict[pred.item()]}")
                    plt.axis("off")
                    plt.savefig(
                        os.path.join(save_dir, f"{base}_cls_overlay_{case}.png"),
                        bbox_inches="tight", pad_inches=0
                    )
                    plt.close()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "cls_attention_results.csv"), index=False)

    print(f"Saved CLS attention visuals & CSV to: {save_dir}")
    total = len(df)
    bal_acc = balanced_accuracy_score(df["true_label"], df["predicted_label"])
    print(f"Total samples: {total}, Balanced Acc: {bal_acc:.2%}")

    inside = df["inside_cls"]
    if inside.any():
        acc_in = (df[inside]["true_label"] == df[inside]["predicted_label"]).mean()
        print(f"CLS INSIDE bbox acc: {acc_in:.2%}")
    if (~inside).any():
        acc_out = (df[~inside]["true_label"] == df[~inside]["predicted_label"]).mean()
        print(f"CLS OUTSIDE bbox acc: {acc_out:.2%}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification Training with attention pretraining')
    parser.add_argument('--dataset_path', type=str, default='ds_sun_10',#ds_sun_10_pnp', #'ds_sun_10', # unbalanced nell'ultimo modello
                        help='Path to the dataset folder (with CSV splits)')
    parser.add_argument('--store_name', type=str, default='PROVA', help='Model storing name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate') # krenzen 0.00016 our 1e-7
    parser.add_argument('--lr_scale', type=float, default=0.1, help='LR reduction factor')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay') #from -6
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    #parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--short_training', action='store_true', help='Enable short training')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--only_test', action='store_true', help='Only test the model')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience') # 30
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank or block expansion grouping size')
    parser.add_argument('--variant', type=str, default='full', help='Variant: "lora" or "block" (or "full")')
    parser.add_argument('--split', type=int, default=None, help='Split to use for block expansion')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of pretraining epochs')
    parser.add_argument('--enable_pretraining', action='store_true', help='Enable pretraining phase using DINOLoss')
    parser.add_argument('--drop', action='store_true', help='drop 2 cases from the dataset')
    #parser.add_argument('--use_rollout', action='store_true', help='Use attention rollout for pretraining overlap loss')
    parser.add_argument('--model_name', type=str, default='vit-l32-224-in21k',
                        help='Model name for classification (e.g., "vit-b16-224-in21k")')
    # make me choose if 2 or 3 classes
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification')
    # inside_weight=1.0, outside_weight=1.0
    parser.add_argument('--penalize_outside', action='store_true', help='Penalize outside-bbox attention in Overlap Loss')
    parser.add_argument('--inside_weight', type=float, default=1.0, help='Weight for inside-bbox attention in Overlap Loss')
    parser.add_argument('--outside_weight', type=float, default=1.0, help='Weight for outside-bbox attention in Overlap Loss')



    args = parser.parse_args()

    # set classes based on num_classes
    if args.num_classes == 2:
        from utils import histology_to_int_dict_2classes as histology_to_int_dict
        from utils import int_to_histology_dict_2classes as int_to_histology_dict
        args.dataset_path = 'ds_sun_10_pnp'
    if args.num_classes == 3:
        from utils import histology_to_int_dict_3classes as histology_to_int_dict
        from utils import int_to_histology_dict_3classes as int_to_histology_dict
        args.dataset_path = 'ds_sun_10'
        
    # Create store directory if it doesn't exist
    store_path = os.path.join(os.getcwd(), args.store_name)
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # Set dataset paths for classification and pretraining
    

    train_dataset_path = os.path.join(os.getcwd(), args.dataset_path, 'train_dataset_sun.csv')
    val_dataset_path = os.path.join(os.getcwd(), args.dataset_path, 'val_dataset_sun.csv')
    test_dataset_path = os.path.join(os.getcwd(), args.dataset_path, 'test_dataset_sun.csv')

    # If pretraining is enabled, create a pretraining dataset and dataloader using PretrainDataset.
    if args.enable_pretraining:
        print("Using public pretraining dataset")
        pretrain_dataset = BBoxTeacherStudentDataset_SUN(train_dataset_path, pretraining=True, drop=args.drop, num_classes=args.num_classes)
        pretrain_val_dataset = BBoxTeacherStudentDataset_SUN(val_dataset_path, test=True, num_classes=args.num_classes)
        # use only 10 samples for pretraining
        if args.short_training:
            pretrain_dataset = torch.utils.data.Subset(pretrain_dataset,  indices=range(min(100, len(pretrain_dataset))))  # Creates list of indices [0,1,2,...,999]

        pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True,
                                                          num_workers=args.num_workers)
        pretrain_val_dataloader = torch.utils.data.DataLoader(pretrain_val_dataset, batch_size=args.batch_size, shuffle=False,
                                                              num_workers=args.num_workers)
        pretrain_dataset = BBoxTeacherStudentDataset_SUN(train_dataset_path, pretraining=True, drop=args.drop, num_classes=args.num_classes)
        pretrain_val_dataset = BBoxTeacherStudentDataset_SUN(val_dataset_path, test=True, num_classes=args.num_classes)
        

    print('Using BBox Teacher-Student Dataset for classification')
    print(f'Train dataset loading from: {train_dataset_path}')
    train_dataset = BBoxTeacherStudentDataset_SUN(train_dataset_path, test=False, num_classes=args.num_classes)
    print(f'Validation dataset loading from: {val_dataset_path}')
    val_dataset = BBoxTeacherStudentDataset_SUN(val_dataset_path, test=True, num_classes=args.num_classes)
    print(f'Test dataset loading from: {test_dataset_path}')
    test_dataset = BBoxTeacherStudentDataset_SUN(test_dataset_path, test=True, num_classes=args.num_classes)
    num_classes = train_dataset.num_classes()



    # Compute class weights and define loss function
    class_weights = train_dataset.compute_class_weights()
    ce_loss_fn = nn.CrossEntropyLoss()#weight=class_weights.to(args.device))

    if args.short_training:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(len(train_dataset) // 100))
        val_dataset = torch.utils.data.Subset(val_dataset, np.arange(len(val_dataset) // 100))
        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(len(test_dataset) // 100))

    # Set up DataLoaders for classification
    print('Preparing DataLoaders for classification')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers)

    if args.device == 'cuda':
        torch.cuda.init()


    model_name = args.model_name
    print(f'Using model: {model_name}')
    
    # If pretraining is enabled, perform the pretraining phase
    if args.enable_pretraining:
        print('Pretraining phase enabled')
        
        student_model = ClassificationModel(
            model_name=model_name,
            optimizer="adamw",
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler="cosine",
            warmup_steps=0,
            n_classes=num_classes,
            image_size=224,
            training_mode="full",
            device='cuda',
            from_scratch=False,
        )
        student_model.net.set_attn_implementation("eager")
        student_model.net.config.output_attentions = True


        
        student_model.to(args.device)
        pretrain_optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pretrain_optimizer, mode='min', factor=args.lr_scale, patience=6, verbose=True, min_lr=1e-8, threshold=1e-4)
    
    
        print('Using Attention Overlap Loss pretraining')
        overlap_loss_fn = AttentionOverlapLoss(target_size=(224, 224), penalize_outside=args.penalize_outside, inside_weight=args.inside_weight, outside_weight=args.outside_weight)
        pretrain_bbox_classifier_overlap(args, student_model, pretrain_dataloader, pretrain_val_dataloader, pretrain_optimizer, pretrain_scheduler, overlap_loss_fn)
        state_dict_path = os.path.join(store_path, 'best_model_pretrain_overlap_only.pth')

        student_model.load_state_dict(torch.load(state_dict_path), strict=False)
        
        print('Pretrained model loaded.')

        

    # if not pretraining, create the student model directly
    else:
        print('Pretraining phase not enabled, creating student model directly')
        
        student_model = ClassificationModel(
            model_name=model_name,
            optimizer="adamw",
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler="cosine",
            warmup_steps=0,
            n_classes=num_classes,
            image_size=224,
            
            
            from_scratch=False,
        )
        student_model.net.set_attn_implementation("eager")
        student_model.net.config.output_attentions = True

    # Save model architecture to file
    with open(os.path.join(store_path, 'model_architecture.txt'), 'w') as f:
        f.write(str(student_model))
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params}, Trainable Parameters: {trainable_params}')
    print('Number of classes:', num_classes)

    student_model.to(args.device)
        
    # if variant is 'lora' and pretraining is enabled, load the pretrained model and apply LoRA
    if args.variant == 'lora' and args.enable_pretraining:
        print('Loading pretrained model with LoRA')
        student_model = load_pretrained_and_apply_lora(
            pretrained_path=os.path.join(store_path, 'best_model_pretrain.pth'),
            args=args,
            num_classes=num_classes,
            device=args.device
        )

    elif args.variant == 'lora' and not args.enable_pretraining:
        print('Loading pretrained model with LoRA')
        student_model = load_pretrained_and_apply_lora(
            pretrained_path=None,
            args=args,
            num_classes=num_classes,
            device=args.device
        )

  
        

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scale, patience=6, verbose=True, min_lr=1e-8, threshold=1e-4)
    # set a cosine scheduler
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * args.lr_scale)


    if not args.only_test:
        if args.resume:
            print('Resuming training')
            resume_path = os.path.join(store_path, 'final_model.pth')
            student_state_dict = torch.load(resume_path, map_location=args.device)
            # Remove DataParallel prefix if present
            student_state_dict = {k.replace('module.', ''): v for k, v in student_state_dict.items()}
            student_model.load_state_dict(student_state_dict, strict=True)
            print('Student model loaded.')
        print('Starting Classification training')
        print('Trainable parameters:', sum(p.numel() for p in student_model.parameters() if p.requires_grad))
        print('Total parameters:', sum(p.numel() for p in student_model.parameters()))
        train_bbox_classifier(args, student_model, train_dataloader, val_dataloader,
                              optimizer, scheduler, ce_loss_fn)
        print('Training complete')

    # Reload best student model for testing
    
    student_model = ClassificationModel(
        model_name=model_name,
        optimizer="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler="cosine",
        warmup_steps=0,
        n_classes=num_classes,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        label_smoothing=0.0,
        image_size=224,
        weights=None,
        training_mode=args.variant,  # "full", "lora", or "block_expansion" (or "none")
        lora_r=args.rank,
        lora_alpha=1,
        lora_target_modules=["query", "key", "value"],
        lora_dropout=0.3,
        lora_bias="none",
        split=args.split,
        
        from_scratch=False,
    )
    student_model.net.set_attn_implementation("eager")
    student_model.net.config.output_attentions = True



    best_model_path = os.path.join(store_path, 'best_model.pth')
    
    student_model.load_state_dict(torch.load(best_model_path), strict=True)
    student_model.to(args.device)


    
    


    # Wrap the testing model too if multiple GPUs are available
    if args.device == 'cuda' and torch.cuda.device_count() > 1:
        student_model = nn.DataParallel(student_model)
    
    test_model(args, student_model, test_dataloader, ce_loss_fn, num_classes=num_classes)

    





    print('Testing complete')

    # Save args as JSON file in the output folder
    args_json_path = os.path.join(store_path, 'args.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_json_path}")


    # Save attention masks and images
    save_attention_masks_and_images_NEW(args, student_model, test_dataloader)
