
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="🏥 Medical AI Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-success {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-warning {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture
class MedicalPretrainedCNN(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet50"):
        super(MedicalPretrainedCNN, self).__init__()
        
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Try different model paths
        model_paths = [
            "pneumonia_model_deployment.pth",
            "/content/pneumonia_model_deployment.pth",
            "/content/drive/MyDrive/pneumonia_model_deployment.pth"
        ]
        
        model = None
        model_info = {}
        
        for path in model_paths:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                
                if "model_config" in checkpoint:
                    config = checkpoint["model_config"]
                else:
                    config = {"backbone": "resnet50", "num_classes": 2}
                
                model = MedicalPretrainedCNN(
                    num_classes=config.get("num_classes", 2),
                    backbone=config.get("backbone", "resnet50")
                )
                
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                
                model_info = {
                    "backbone": config.get("backbone", "resnet50"),
                    "accuracy": checkpoint.get("performance", {}).get("test_accuracy", 
                                            checkpoint.get("best_val_acc", 0.85)),
                    "loaded_from": path
                }
                
                st.success(f"✅ Model loaded from: {path}")
                break
                
            except Exception as e:
                continue
        
        if model is None:
            st.error("❌ No model file found. Please upload your trained model.")
            st.info("Expected files: pneumonia_model_deployment.pth")
            return None, None
            
        return model, model_info
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

@st.cache_data
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, transform):
    """Make prediction on image"""
    try:
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "predicted_class": predicted_class,
            "predicted_label": ["Normal", "Pneumonia"][predicted_class],
            "confidence": confidence,
            "probabilities": probabilities[0].numpy()
        }
        
    except Exception as e:
        return {"error": str(e)}

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical AI Diagnosis Assistant</h1>
        <p>AI-powered chest X-ray analysis for pneumonia detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_info = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("📋 Model Information")
        if model_info:
            st.info(f"""
            **Architecture:** {model_info.get("backbone", "Unknown")}
            **Accuracy:** {model_info.get("accuracy", 0):.1%}
            **Status:** ✅ Ready
            """)
        
        st.header("📖 Instructions")
        st.markdown("""
        1. Upload a chest X-ray image
        2. Wait for AI analysis
        3. Review diagnosis results
        4. Download report if needed
        
        ⚠️ **For educational use only**
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose chest X-ray image...",
            type=["png", "jpg", "jpeg"],
            help="Upload clear chest X-ray for analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            if st.button("🧠 Analyze Image", type="primary"):
                with st.spinner("🔬 AI analyzing..."):
                    transform = get_transforms()
                    result = predict_image(model, image, transform)
                    
                    if "error" in result:
                        st.error(f"Analysis failed: {result['error']}")
                    else:
                        st.session_state.prediction = result
    
    with col2:
        st.header("📊 Analysis Results")
        
        if hasattr(st.session_state, "prediction"):
            result = st.session_state.prediction
            confidence_pct = result["confidence"] * 100
            
            # Main prediction
            if result["predicted_class"] == 0:
                st.markdown(f"""
                <div class="prediction-success">
                    <h2>✅ Diagnosis: {result["predicted_label"]}</h2>
                    <h3>Confidence: {confidence_pct:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-warning">
                    <h2>⚠️ Diagnosis: {result["predicted_label"]}</h2>
                    <h3>Confidence: {confidence_pct:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_pct,
                title={"text": "Confidence Level"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 60], "color": "lightgray"},
                        {"range": [60, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Probabilities
            st.subheader("📊 Class Probabilities")
            prob_data = {
                "Class": ["Normal", "Pneumonia"],
                "Probability": [result["probabilities"][0], result["probabilities"][1]]
            }
            
            fig = px.bar(x=prob_data["Class"], y=prob_data["Probability"], 
                        color=prob_data["Probability"], color_continuous_scale="RdYlGn_r")
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("📋 Clinical Recommendations")
            
            if result["predicted_class"] == 0:
                st.success("""
                **✅ Low Risk Assessment**
                - No pneumonia detected
                - Continue routine monitoring
                - Normal X-ray pattern
                """)
            else:
                st.warning("""
                **⚠️ Abnormality Detected**
                - Signs consistent with pneumonia
                - Medical evaluation recommended
                - Consult radiologist
                """)
            
            # Download report
            if st.button("📄 Generate Report"):
                report = f"""
MEDICAL AI DIAGNOSIS REPORT
==========================

Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Diagnosis: {result["predicted_label"]}
Confidence: {confidence_pct:.2f}%

Probabilities:
- Normal: {result["probabilities"][0]:.3f}
- Pneumonia: {result["probabilities"][1]:.3f}

Risk Assessment: {"Low Risk" if result["predicted_class"] == 0 else "Requires Attention"}

Disclaimer: For educational use only. Consult medical professionals.
                """
                
                st.download_button(
                    "📥 Download Report",
                    report,
                    file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("👆 Upload an image to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>🏥 Medical AI Diagnosis Assistant</strong></p>
        <p>For Educational/Research Use Only • Always Consult Medical Professionals</p>
        <small>Running on Google Colab • Powered by PyTorch & Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
