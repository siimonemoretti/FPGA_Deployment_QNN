# evaluation_utils.py

import torch
import csv
import os

def evaluate_and_save_csv(model, test_loader, classes, device, batch_size, epoch, optimizer, criterion):
   model.eval()
   with torch.no_grad():
      n_correct = 0
      n_samples = 0
      n_class_correct = [0 for _ in range(10)]
      n_class_samples = [0 for _ in range(10)]

      for images, labels in test_loader:
         images = images.to(device)
         labels = labels.to(device)
         outputs = model(images)
         _, predicted = torch.max(outputs, 1)
         n_samples += labels.size(0)
         n_correct += (predicted == labels).sum().item()

         for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
               n_class_correct[label] += 1
            n_class_samples[label] += 1

      overall_accuracy = 100.0 * n_correct / n_samples
      print(f'Accuracy of the network: {overall_accuracy:.2f} %')

      class_accuracies = []
      for i in range(10):
         acc = 100.0 * n_class_correct[i] / n_class_samples[i]
         print(f'Accuracy of {classes[i]}: {acc:.2f} %')
         class_accuracies.append((classes[i], acc))

      # Ensure results directory exists
      os.makedirs("../results", exist_ok=True)
      csv_file = "../results/QAT.csv"
      write_header = not os.path.exists(csv_file)

      with open(csv_file, mode='a', newline='') as file:
         writer = csv.writer(file)

         if write_header:
            header = [
               "Epoch", "Batch Size", "Learning Rate", "Momentum",
               "Loss Function", "Optimizer", "Overall Accuracy"
            ] + [f"{cls} Accuracy" for cls in classes]
            writer.writerow(header)

         row = [
            epoch + 1,
            batch_size,
            optimizer.param_groups[0]['lr'],
            optimizer.param_groups[0].get('momentum', 'N/A'),
            criterion.__class__.__name__,
            optimizer.__class__.__name__,
            f"{overall_accuracy:.2f}"
         ] + [f"{acc:.2f}" for _, acc in class_accuracies]

         writer.writerow(row)
