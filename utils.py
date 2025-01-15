import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0)}")

def analyze_sensitivity(self, input_data, delta=0.01):
        input_tensor = torch.tensor(input_data, device=device).float()
        base_output = self.process_input_batch(input_tensor.unsqueeze(0))
        sensitivities = []
        for i in range(len(input_tensor.flatten())):
            perturbed_input = input_tensor.clone().flatten()
            perturbed_input[i] += delta
            new_output = self.process_input_batch(perturbed_input.view(1, -1))
            sensitivity = (new_output - base_output) / delta
            sensitivities.append((i, sensitivity))
        return sensitivities

def plot_learning_curves(self):
    plt.figure()
    plt.plot(self.training_errors, label='Training Error')
    if len(self.validation_errors) > 0:
        plt.plot(self.validation_errors, label='Validation Error', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()