import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

class Config:
    CLIP_LENGTH = 16
    FRAME_SIZE = (224, 224)
    BATCH_SIZE = 8
    
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 50
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_ROOT = 'path/to/dataset'
    MODEL_SAVE_PATH = 'saved_models'

cfg = Config()

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        if mode == 'train':
            self.video_paths = self._get_video_paths(os.path.join(root_dir, 'train', 'normal'))
            self.labels = [0] * len(self.video_paths)
        else:
            normal_paths = self._get_video_paths(os.path.join(root_dir, 'test', 'normal'))
            abnormal_paths = self._get_video_paths(os.path.join(root_dir, 'test', 'abnormal'))
            self.video_paths = normal_paths + abnormal_paths
            self.labels = [0] * len(normal_paths) + [1] * len(abnormal_paths)
    
    def _get_video_paths(self, directory):
        if not os.path.exists(directory):
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if f.endswith(('.mp4', '.avi', '.mov'))]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self._load_video(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        clips = self._frames_to_clips(frames)
        
        return clips, label
    
    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, cfg.FRAME_SIZE)
            frames.append(frame)
        
        cap.release()
        
        while len(frames) < cfg.CLIP_LENGTH:
            if len(frames) == 0:
                empty_frame = np.zeros((cfg.FRAME_SIZE[0], cfg.FRAME_SIZE[1], 3), dtype=np.uint8)
                frames.append(empty_frame)
            else:
                frames.append(frames[-1])
                
        return frames
    
    def _frames_to_clips(self, frames):
        if len(frames) > cfg.CLIP_LENGTH:
            indices = np.linspace(0, len(frames)-1, cfg.CLIP_LENGTH, dtype=int)
            frames = [frames[i] for i in indices]
        
        clip = torch.stack(frames)
        
        return clip

class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        
        self.feature_extractor = models.video.r3d_18(pretrained=True)
        feature_dim = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=cfg.HIDDEN_SIZE,
            num_layers=cfg.NUM_LAYERS,
            batch_first=True,
            dropout=cfg.DROPOUT if cfg.NUM_LAYERS > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(cfg.HIDDEN_SIZE * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg.HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, clip_length, c, h, w = x.shape
        
        x = x.permute(0, 2, 1, 3, 4)
        
        features = self.feature_extractor(x)
        
        features = features.unsqueeze(1)
        
        lstm_out, _ = self.lstm(features)
        
        attn_weights = self.attention(lstm_out)
        context = torch.bmm(attn_weights.permute(0, 2, 1), lstm_out)
        context = context.squeeze(1)
        
        score = self.classifier(context)
        
        return score

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for clips, labels in tqdm(train_loader, desc="Training"):
        clips = clips.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        outputs = model(clips)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for clips, labels in tqdm(test_loader, desc="Evaluating"):
            clips = clips.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(clips)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            all_scores.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_scores = np.array(all_scores).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    auc_score = roc_auc_score(all_labels, all_scores)
    
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return total_loss / len(test_loader), auc_score, best_threshold, best_f1

def train_model(train_dataset, test_dataset, cfg):
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)
    
    model = AnomalyDetector().to(cfg.DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)
    
    best_auc = 0
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        
        val_loss, auc_score, threshold, f1_score = evaluate(model, test_loader, criterion, cfg.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"AUC: {auc_score:.4f}, "
              f"Best F1: {f1_score:.4f} (threshold={threshold:.2f})")
        
        if auc_score > best_auc:
            best_auc = auc_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': threshold,
                'auc': auc_score,
                'f1': f1_score
            }, os.path.join(cfg.MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"Model saved with AUC: {auc_score:.4f}")
    
    return model

def visualize_anomalies(model, video_path, threshold, device):
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    anomaly_scores = []
    
    frames_buffer = []
    frame_indices = []
    
    with torch.no_grad():
        for frame_idx in tqdm(range(frame_count), desc="Processing video"):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, cfg.FRAME_SIZE)
            
            frame_tensor = transform(frame_resized)
            
            frames_buffer.append(frame_tensor)
            frame_indices.append(frame_idx)
            
            if len(frames_buffer) == cfg.CLIP_LENGTH:
                clip = torch.stack(frames_buffer).unsqueeze(0).to(device)
                
                score = model(clip).cpu().numpy()[0, 0]
                
                for _ in range(1):
                    anomaly_scores.append(score)
                    
                frames_buffer = frames_buffer[1:]
                frame_indices = frame_indices[1:]
    
    cap.release()
    
    while len(anomaly_scores) < frame_count:
        if len(anomaly_scores) > 0:
            anomaly_scores.append(anomaly_scores[-1])
        else:
            anomaly_scores.append(0)
    
    plt.figure(figsize=(12, 6))
    
    time_axis = np.arange(len(anomaly_scores)) / fps
    plt.plot(time_axis, anomaly_scores)
    
    anomaly_regions = np.array(anomaly_scores) > threshold
    plt.fill_between(time_axis, 0, 1, where=anomaly_regions, color='red', alpha=0.3)
    
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Anomaly Score')
    plt.title('Abnormal Behavior Detection Results')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    plt.savefig('anomaly_detection_results.png')
    plt.close()
    
    return anomaly_scores, time_axis

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = VideoDataset(cfg.DATA_ROOT, mode='train', transform=transform)
    test_dataset = VideoDataset(cfg.DATA_ROOT, mode='test', transform=transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    model = train_model(train_dataset, test_dataset, cfg)
    
    checkpoint = torch.load(os.path.join(cfg.MODEL_SAVE_PATH, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    threshold = checkpoint['threshold']
    
    test_video_path = os.path.join(cfg.DATA_ROOT, 'test', 'abnormal', 'example.mp4')
    anomaly_scores, time_axis = visualize_anomalies(model, test_video_path, threshold, cfg.DEVICE)
    
    print("异常行为检测完成！")

if __name__ == "__main__":
    main()