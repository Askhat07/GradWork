# import torch
# from torchvision import transforms
# from PIL import Image
#
# # Путь к сохранённой модели
# model_path = 'C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/plant-disease-model-complete.pth'
#
# # Загрузка модели
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load(model_path, map_location=device)
# model.eval()
#
# # Список классов (можно загрузить из train.classes, если сохранил)
# classes = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
#            # добавь все свои классы...
#           ]
#
# # Функция для предобработки изображения
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     return transform(image)
#
# # Функция для предсказания
# def predict_image(image_path):
#     image = preprocess_image(image_path)
#     image = image.unsqueeze(0).to(device)
#     outputs = model(image)
#     _, predicted = torch.max(outputs, 1)
#     predicted_class = classes[predicted.item()]
#     return predicted_class
#
#
# if __name__ == '__main__':
#     # Пример использования
#     test_image_path = 'C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/Plant Diseases Dataset/train/Apple___Black_rot/00e909aa-e3ae-4558-9961-336bb0f35db3___JR_FrgE.S 8593_270deg.JPG'
#     result = predict_image(test_image_path)
#     print(f"Predicted class: {result}")

# import torch
# from torchvision import transforms
# from PIL import Image
# from main import res_net_9  # Импортируем модель из основного файла
#
# # Загрузка обученной модели
# MODEL_PATH = "C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/plant-disease-model-complete.pth"  # Путь к сохраненной модели
# model = res_net_9(num_classes=5)  # Укажи правильное количество классов
# model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
# model.eval()
#
# # Функция для предобработки изображения
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),  # Приведение к размеру, использованному при обучении
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     image = Image.open(image_path).convert("RGB")
#     return transform(image).unsqueeze(0)  # Добавляем batch dimension
#
# # Функция для предсказания заболевания
# def predict(image_path):
#     image_tensor = preprocess_image(image_path)
#     with torch.no_grad():
#         output = model(image_tensor)
#         predicted_class = torch.argmax(output, dim=1).item()
#     return predicted_class
#
# if __name__ == "__main__":
#     image_path = input("C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/Plant Diseases Dataset/train/Apple___Black_rot/00e909aa-e3ae-4558-9961-336bb0f35db3___JR_FrgE.S 8593_270deg.JPG")
#     prediction = predict(image_path)
#     print(f"Предсказанный класс: {prediction}")

# import torch
# from torchvision import transforms
# from PIL import Image
# from model import res_net_9  # если у тебя в отдельном файле
# # или скопируй сюда определение res_net_9
#
# # Параметры
# in_channels = 3
# num_classes = 38  # столько у тебя классов, можешь поменять
# model = res_net_9(in_channels, num_classes)
#
# # Загрузка весов
# state_dict_path = 'C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/plant-disease-model.pth'
# model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
# model.eval()
#
# # Подготовка картинки
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])
#
# image_path = 'C:/Users/ashat/Pictures/test_plant.jpg'
# image = Image.open(image_path).convert('RGB')
# image = transform(image).unsqueeze(0)
#
# # Предсказание
# with torch.no_grad():
#     output = model(image)
#     _, predicted = torch.max(output, 1)
#
# print(f"Predicted class index: {predicted.item()}")

import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

weight_path = "C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/plant-disease-model.pth"
class_names = sorted(os.listdir("C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/Plant Diseases Dataset/train"))

print(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(weight_path, num_classes=len(class_names), device=device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return None

    try:
        img = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, dim=1)
            predicted_class = class_names[pred.item()]
        return predicted_class
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None


image_path = "C:/Users/ashat/PycharmProjects/GraduateWork/Plant/Plant Diseases Dataset/test/test/AppleBlackRot.JPG"
prediction = predict_image(image_path)
print(f"Predicted class: {prediction}")