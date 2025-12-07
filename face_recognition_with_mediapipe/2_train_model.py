"""
Model Training Script - AI Without ML Assignment
Trains LBPH Face Recognizer on captured face images
Part 2 of 3: Training the Classical Feature-Based Recognizer
"""
import cv2
import os
import json
import numpy as np
from datetime import datetime


class LBPHTrainer:
    def __init__(self, dataset_dir="datasets", model_dir="models"):
        """
        Initialize LBPH Trainer
        Args:
            dataset_dir: Directory containing captured face images
            model_dir: Directory to save trained model
        """
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"✓ Created model directory: {self.model_dir}")
        
        # Initialize LBPH Face Recognizer
        # LBPH = Local Binary Patterns Histograms
        # This is a classical computer vision algorithm (not ML)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,        # Radius of LBP pattern
            neighbors=8,     # Number of neighbors
            grid_x=8,        # Grid size in X
            grid_y=8,        # Grid size in Y
            threshold=100.0  # Confidence threshold
        )
    
    def load_training_data(self):
        """
        Load training images and labels from dataset directory
        Returns:
            faces: List of face images (grayscale)
            labels: List of corresponding labels (person IDs)
            label_map: Dictionary mapping label IDs to person names
        """
        print(f"\n{'='*60}")
        print("Loading training data...")
        print(f"{'='*60}\n")
        
        if not os.path.exists(self.dataset_dir):
            raise Exception(f"Dataset directory not found: {self.dataset_dir}")
        
        faces = []
        labels = []
        label_map = {}
        current_label = 0
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(self.dataset_dir) 
                      if os.path.isdir(os.path.join(self.dataset_dir, d))]
        
        if not person_dirs:
            raise Exception(f"No person directories found in {self.dataset_dir}")
        
        print(f"Found {len(person_dirs)} person(s) in dataset:\n")
        
        # Process each person's directory
        for person_name in sorted(person_dirs):
            person_path = os.path.join(self.dataset_dir, person_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(person_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"⚠ No images found for {person_name}, skipping...")
                continue
            
            # Assign label to this person
            label_map[current_label] = person_name
            
            print(f"  [{current_label}] {person_name}: {len(image_files)} images")
            
            # Load each image
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                
                # Read image
                img = cv2.imread(image_path)
                
                if img is None:
                    print(f"    ⚠ Failed to load: {image_file}")
                    continue
                
                # Convert to grayscale (LBPH requires grayscale)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize to standard size for consistency
                gray = cv2.resize(gray, (200, 200))
                
                # Add to training data
                faces.append(gray)
                labels.append(current_label)
            
            current_label += 1
        
        print(f"\n{'='*60}")
        print(f"✓ Loaded {len(faces)} images from {len(label_map)} person(s)")
        print(f"{'='*60}\n")
        
        return faces, labels, label_map
    
    def train(self):
        """
        Train LBPH Face Recognizer and save model
        """
        print(f"\n{'='*60}")
        print("Training LBPH Face Recognizer")
        print(f"{'='*60}\n")
        
        # Load training data
        faces, labels, label_map = self.load_training_data()
        
        if len(faces) == 0:
            raise Exception("No training data found!")
        
        # Convert to numpy arrays
        faces = np.array(faces)
        labels = np.array(labels)
        
        print("Training model...")
        print(f"  - Algorithm: LBPH (Local Binary Patterns Histograms)")
        print(f"  - Training samples: {len(faces)}")
        print(f"  - Number of classes: {len(label_map)}")
        print(f"  - Image size: {faces[0].shape}")
        print()
        
        # Train the recognizer
        start_time = datetime.now()
        self.recognizer.train(faces, labels)
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Save the model
        model_path = os.path.join(self.model_dir, "lbph_face_recognizer.yml")
        self.recognizer.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save label map
        label_map_path = os.path.join(self.model_dir, "label_map.json")
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        print(f"✓ Label map saved to: {label_map_path}")
        
        # Display training summary
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}\n")
        print(f"Total training images: {len(faces)}")
        print(f"Number of people: {len(label_map)}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"\nPeople in model:")
        for label_id, name in sorted(label_map.items(), key=lambda x: int(x[0])):
            count = np.sum(labels == int(label_id))
            print(f"  [{label_id}] {name}: {count} images")
        
        print(f"\n{'='*60}")
        print("✓ Model training complete!")
        print("Next step: Run '3_predict_faces.py' for face recognition")
        print(f"{'='*60}\n")


def main():
    """Main function to train the model"""
    print("="*60)
    print("Model Training Script - AI Without ML Assignment")
    print("Part 2/3: Training LBPH Face Recognizer")
    print("="*60)
    
    try:
        # Initialize trainer
        trainer = LBPHTrainer(dataset_dir="datasets", model_dir="models")
        
        # Train the model
        trainer.train()
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Captured face images using '1_capture_faces.py'")
        print("  2. Installed opencv-contrib-python: pip install opencv-contrib-python")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
