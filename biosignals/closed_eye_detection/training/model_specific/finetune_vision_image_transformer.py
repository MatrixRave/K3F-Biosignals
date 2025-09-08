import warnings  # Import the 'warnings' module for handling warnings
warnings.filterwarnings("ignore")  # Ignore warnings during execution

import evaluate  # Import the 'evaluate' module
from datasets import Dataset, Image, ClassLabel  # Import custom 'Dataset', 'ClassLabel', and 'Image' classes
from transformers import (  # Import various modules from the Transformers library
    TrainingArguments,  # For training arguments
    Trainer,  # For model training
    ViTImageProcessor,  # For processing image data with ViT models
    ViTForImageClassification,  # ViT model for image classification
    DefaultDataCollator  # For collating data in the default way
)
import torch  # Import PyTorch for deep learning
from torchvision.transforms import (  # Import image transformation functions
    CenterCrop,  # Center crop an image
    Compose,  # Compose multiple image transformations
    Normalize,  # Normalize image pixel values
    RandomRotation,  # Apply random rotation to images
    RandomResizedCrop,  # Crop and resize images randomly
    RandomHorizontalFlip,  # Apply random horizontal flip
    RandomAdjustSharpness,  # Adjust sharpness randomly
    Resize,  # Resize images
    ToTensor  # Convert images to PyTorch tensors
)

from dataloader import load_image_filepaths_as_dataframe, convert_image_to_greyscale

import pprint
import os

import mlflow


def finetune(ds_dir, ):

    ds_dir = '/home/nive1002/mEBAL2-dataset/Blinks-Unblinks_prepared_greyscale'
    ds_dir = '/home/nive1002/mEBAL2-dataset/Blinks-Unblinks_prepared_filtered'

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8877")

    model_str = 'google/vit-base-patch16-224-in21k' # 'dima806/closed_eyes_image_detection'  # 'google/vit-base-patch16-224-in21k'

    model_name = "vit-base-patch16-224-in21k_ft_closed_eyes_detection_greyscale_filtered_v1"

    mlflow.set_experiment("Closed Eye Detection")

    mlflow.set_tag("Training Info", "VIT model finetuned on greyscale images for closed eyes detection")
    mlflow.set_tag("Images", "greyscale mEBAL2, filtered v1")
    mlflow.set_tag("dataset dir", ds_dir)
    mlflow.set_tag("base model", model_str)
    mlflow.set_tag("model name", model_name)



    open_closed_eye_ds_df = load_image_filepaths_as_dataframe(ds_dir)
    dataset = Dataset.from_pandas(open_closed_eye_ds_df)

    dataset = dataset.map(convert_image_to_greyscale)
    dataset = dataset.cast_column("image", Image())
    print(dataset)

    #mlflow.log_input(dataset, context="training")

    labels_list = ['open', 'closed']

    # Initialize empty dictionaries to map labels to IDs and vice versa
    label2id, id2label = dict(), dict()

    # Iterate over the unique labels and assign each label an ID, and vice versa
    for i, label in enumerate(labels_list):
        label2id[label] = i  # Map the label to its corresponding ID
        id2label[i] = label  # Map the ID to its corresponding label

    # Print the resulting dictionaries for reference
    print("Mapping of IDs to Labels:", id2label)
    print("Mapping of Labels to IDs:", label2id)

    # Creating classlabels to match labels to IDs
    classLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

    # Mapping labels to IDs
    def map_label2id(example):
        example['label'] = classLabels.str2int(example['label'])
        return example

    dataset = dataset.map(map_label2id, batched=True)

    # Casting label column to ClassLabel Object
    dataset = dataset.cast_column('label', classLabels)

    # Splitting the dataset into training and testing sets using an 90-10 split ratio.
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column="label")

    # Extracting the training data from the split dataset.
    train_data = dataset['train']

    # Extracting the testing data from the split dataset.
    test_data = dataset['test']

    # Create a processor for ViT model input from the pre-trained model
    processor = ViTImageProcessor.from_pretrained(model_str)

    # Retrieve the image mean and standard deviation used for normalization
    image_mean, image_std = processor.image_mean, processor.image_std

    # Get the size (height) of the ViT model's input images
    size = processor.size["height"]
    print("VIT-model image input size: ", size)

    # Define a normalization transformation for the input images
    normalize = Normalize(mean=image_mean, std=image_std)

    # Define a set of transformations for training data
    _train_transforms = Compose(
        [
            Resize((size, size)),  # Resize images to the ViT model's input size
            RandomRotation(90),  # Apply random rotation
            RandomAdjustSharpness(2),  # Adjust sharpness randomly
            RandomHorizontalFlip(0.5),  # Apply random horizontal flip
            ToTensor(),  # Convert images to tensors
            normalize  # Normalize images using mean and std
        ]
    )

    # Define a set of transformations for validation data
    _val_transforms = Compose(
        [
            Resize((size, size)),  # Resize images to the ViT model's input size
            ToTensor(),  # Convert images to tensors
            normalize  # Normalize images using mean and std
        ]
    )


    # Define a function to apply training transformations to a batch of examples
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples


    # Define a function to apply validation transformations to a batch of examples
    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    # Set the transforms for the training data
    train_data.set_transform(train_transforms)

    # Set the transforms for the test/validation data
    test_data.set_transform(val_transforms)

    # Define a collate function that prepares batched data for model training.
    def collate_fn(examples):
        # Stack the pixel values from individual examples into a single tensor.
        pixel_values = torch.stack([example["pixel_values"] for example in examples])

        # Convert the label strings in examples to corresponding numeric IDs using label2id dictionary.
        labels = torch.tensor([example['label'] for example in examples])

        # Return a dictionary containing the batched pixel values and labels.
        return {"pixel_values": pixel_values, "labels": labels}#


    # Create a ViTForImageClassification model from a pretrained checkpoint with a specified number of output labels.
    model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))

    # Configure the mapping of class labels to their corresponding indices for later reference.
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Calculate and print the number of trainable parameters in millions for the model.
    print("number of trainable parameters in millions:", model.num_parameters(only_trainable=True) / 1e6)

    # Load the accuracy metric from a module named 'evaluate'
    accuracy = evaluate.load("accuracy")

    # Define a function 'compute_metrics' to calculate evaluation metrics
    def compute_metrics(eval_pred):
        # Extract model predictions from the evaluation prediction object
        predictions = eval_pred.predictions

        # Extract true labels from the evaluation prediction object
        label_ids = eval_pred.label_ids

        # Calculate accuracy using the loaded accuracy metric
        # Convert model predictions to class labels by selecting the class with the highest probability (argmax)
        predicted_labels = predictions.argmax(axis=1)

        # Calculate accuracy score by comparing predicted labels to true labels
        acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

        # Return the computed accuracy as a dictionary with the key "accuracy"
        return {
            "accuracy": acc_score
        }

    # Define the name of the evaluation metric to be used during training and evaluation.
    metric_name = "accuracy"

    # Define the number of training epochs for the model.
    num_train_epochs = 50 #50

    # Create an instance of TrainingArguments to configure training settings.
    args = TrainingArguments(
        # Specify the directory where model checkpoints and outputs will be saved.
        output_dir=os.path.join("model", model_name),

        #run_name="", # TODO

        # Specify the directory where training logs will be stored.
        logging_dir='./logs',

        logging_strategy='steps',

        logging_first_step=True,

        logging_steps=64,

        # Define the evaluation strategy, which is performed at the end of each epoch.
        eval_strategy="epoch",

        # Set the learning rate for the optimizer.
        learning_rate=1e-6,

        # Define the batch size for training on each device.
        per_device_train_batch_size=64,

        # Define the batch size for evaluation on each device.
        per_device_eval_batch_size=32,

        # Specify the total number of training epochs.
        num_train_epochs=num_train_epochs,

        # Apply weight decay to prevent overfitting.
        weight_decay=0.02,

        # Set the number of warm-up steps for the learning rate scheduler.
        warmup_steps=50,

        # Disable the removal of unused columns from the dataset.
        remove_unused_columns=False,

        # Define the strategy for saving model checkpoints (per epoch in this case).
        save_strategy='epoch',

        # Load the best model at the end of training.
        load_best_model_at_end=True,

        # Limit the total number of saved checkpoints to save space.
        save_total_limit=2,

        # Specify that training progress should not be reported.
        report_to="mlflow"  # log to none
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    print("Evaluating, before training...")
    pprint.pprint(trainer.evaluate())

    print("Starting training...")


    #with mlflow.start_run() as run:
    trainer.train()

    print("Evaluating, after training...")

    pprint.pprint(trainer.evaluate())

    print("Predict on test data...")

    # Use the trained 'trainer' to make predictions on the 'test_data'.
    outputs = trainer.predict(test_data)

    # Print the metrics obtained from the prediction outputs.
    pprint.pprint(outputs.metrics)

    # Save best model
    trainer.save_model(os.path.join("model", model_name))

    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model},
        task="image-classification",
        artifact_path="fine_tuned_model",
        model_config=trainer.model.config,
        processor=processor,
    )




if __name__ == "__main__":
    finetune()