import pickle

with open("models/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

message = input("Enter email message: ")

prediction = model.predict([message])

print("Prediction:", prediction[0])