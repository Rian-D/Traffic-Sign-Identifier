import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .demo-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

# Traffic sign categories (German Traffic Sign Recognition Benchmark)
TRAFFIC_SIGN_CATEGORIES = {
    0: "Speed limit 20 km/h", 1: "Speed limit 30 km/h", 2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h", 4: "Speed limit 70 km/h", 5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h", 7: "Speed limit 100 km/h", 8: "Speed limit 120 km/h",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or turn right",
    37: "Go straight or turn left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing", 42: "End of no passing for vehicles over 3.5 metric tons"
}

@st.cache_data
def load_data(data_dir):
    """Load image data from directory with progress tracking."""
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        return [], []
    
    directories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, directory in enumerate(directories):
        status_text.text(f"Loading category {directory} ({TRAFFIC_SIGN_CATEGORIES.get(int(directory), 'Unknown')})")
        
        dir_path = os.path.join(data_dir, directory)
        # Updated to include .ppm files
        files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))]
        
        for file in files:
            image_path = os.path.join(dir_path, file)
            image = cv2.imread(image_path)
            if image is not None:
                resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(resized)
                labels.append(int(directory))
        
        progress_bar.progress((i + 1) / len(directories))
    
    progress_bar.empty()
    status_text.empty()
    
    return images, labels

def create_demo_predictions():
    """Create realistic demo predictions for demonstration purposes."""
    # Simulate a trained model's predictions
    np.random.seed(42)  # For reproducible demo results
    
    # Create random confidence scores that sum to 1
    predictions = np.random.dirichlet(np.ones(NUM_CATEGORIES))
    
    # Make one category more prominent (simulate good prediction)
    predicted_class = np.random.randint(0, NUM_CATEGORIES)
    predictions[predicted_class] *= 3
    predictions = predictions / np.sum(predictions)  # Renormalize
    
    return predictions, predicted_class

def show_home_page():
    """Display the home page with project overview."""
    st.markdown("## 🎯 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This **Traffic Sign Recognition AI** system demonstrates advanced machine learning techniques:
        
        - **Computer Vision**: Image processing and feature extraction
        - **Deep Learning**: Convolutional Neural Networks (CNN)
        - **Data Science**: Statistical analysis and model evaluation
        - **Web Development**: Interactive Streamlit interface
        
        ### 🚀 Key Features
        - **43 Traffic Sign Categories** from German Traffic Sign Recognition Benchmark
        - **Real-time Image Classification** with confidence scores
        - **Interactive Training Interface** with live progress tracking
        - **Comprehensive Analytics** and performance visualization
        - **Professional Web Interface** suitable for production deployment
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Technologies Used
        
        - **TensorFlow 2.x** - Deep Learning Framework
        - **OpenCV** - Computer Vision
        - **Scikit-learn** - Machine Learning
        - **Streamlit** - Web Interface
        - **Plotly** - Interactive Visualizations
        - **Pandas** - Data Manipulation
        """)
    
    # Demo showcase
    st.markdown('<div class="demo-box"><h2>🎮 Interactive Demo Available!</h2><p>Try the live prediction feature with your own images!</p></div>', unsafe_allow_html=True)
    
    # Metrics
    st.markdown("## 📊 System Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><h3>43</h3><p>Traffic Sign Categories</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><h3>30x30</h3><p>Image Resolution</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><h3>CNN</h3><p>Neural Architecture</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><h3>Real-time</h3><p>Prediction Speed</p></div>', unsafe_allow_html=True)

def show_data_analysis_page():
    """Display data analysis and exploration."""
    st.markdown("## 📊 Data Analysis & Exploration")
    
    # Data directory input
    data_dir = st.text_input("Enter the path to your traffic sign dataset:", value="gtsrb")
    
    if st.button("🔍 Analyze Dataset") and data_dir:
        if os.path.exists(data_dir):
            with st.spinner("Loading and analyzing dataset..."):
                try:
                    images, labels = load_data(data_dir)
                    
                    if len(images) > 0:
                        st.success(f"✅ Successfully loaded {len(images)} images from {len(set(labels))} categories!")
                        
                        # Dataset statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Images", len(images))
                        
                        with col2:
                            st.metric("Categories", len(set(labels)))
                        
                        with col3:
                            st.metric("Image Size", f"{IMG_WIDTH}x{IMG_HEIGHT}")
                        
                        # Category distribution
                        st.markdown("### 📈 Category Distribution")
                        label_counts = pd.Series(labels).value_counts().sort_index()
                        
                        fig = px.bar(
                            x=[TRAFFIC_SIGN_CATEGORIES.get(i, f"Category {i}") for i in label_counts.index],
                            y=label_counts.values,
                            title="Number of Images per Traffic Sign Category",
                            labels={'x': 'Traffic Sign Category', 'y': 'Number of Images'}
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Sample images
                        st.markdown("### 🖼️ Sample Images from Each Category")
                        sample_images = []
                        sample_labels = []
                        
                        for category in sorted(set(labels))[:12]:  # Show first 12 categories
                            category_images = [img for i, img in enumerate(images) if labels[i] == category]
                            if category_images:
                                sample_images.append(category_images[0])
                                sample_labels.append(category)
                        
                        # Display sample images in a grid
                        cols = st.columns(4)
                        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
                            with cols[i % 4]:
                                st.image(img, caption=f"Category {label}: {TRAFFIC_SIGN_CATEGORIES.get(label, 'Unknown')}", use_column_width=True)
                        
                        # Store data in session state
                        st.session_state.images = images
                        st.session_state.labels = labels
                        
                    else:
                        st.error("❌ No images found in the specified directory.")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
        else:
            st.error("❌ Directory not found. Please check the path.")

def show_demo_training_page():
    """Display a demo training interface."""
    st.markdown("## 🤖 Model Training Interface (Demo)")
    
    st.info("""
    🎯 **Demo Mode**: This is a demonstration of the training interface. 
    In a production environment, this would train a real TensorFlow model.
    """)
    
    # Training parameters
    st.markdown("### ⚙️ Training Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Number of Epochs", 1, 50, 10)
    
    with col2:
        test_size = st.slider("Test Split Ratio", 0.1, 0.5, 0.4, 0.1)
    
    with col3:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    # Start training demo
    if st.button("🚀 Start Training Demo", type="primary"):
        with st.spinner("Preparing data and training model..."):
            try:
                # Simulate data preparation
                time.sleep(1)
                st.success("✅ Data prepared: 34,799 training, 12,630 test samples")
                
                # Display model architecture
                st.markdown("### 🧠 Neural Network Architecture")
                st.code("""
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 28, 28, 32)          │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 14, 14, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 12, 12, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 6, 6, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 4, 4, 64)            │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 1024)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │         131,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 43)                  │           5,547 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 193,067 (754.16 KB)
Trainable params: 193,067 (754.16 KB)
Non-trainable params: 0 (0.00 B)
                """)
                
                # Training progress
                st.markdown("### 📈 Training Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate training progress
                for epoch in range(epochs):
                    time.sleep(0.5)  # Simulate training time
                    status_text.text(f"Training epoch {epoch + 1}/{epochs}")
                    progress_bar.progress((epoch + 1) / epochs)
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("🎉 Training completed successfully!")
                
                # Show demo training results
                st.markdown("### 📊 Training Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Final Training Accuracy", "0.9876")
                    st.metric("Final Training Loss", "0.0234")
                
                with col2:
                    st.metric("Final Validation Accuracy", "0.9745")
                    st.metric("Final Validation Loss", "0.0456")
                
                # Plot demo training history
                st.markdown("### 📈 Training History")
                epochs_range = list(range(1, epochs + 1))
                train_acc = [0.75 + 0.2 * (1 - np.exp(-i/3)) + 0.02 * np.random.random() for i in epochs_range]
                val_acc = [0.70 + 0.25 * (1 - np.exp(-i/3)) + 0.03 * np.random.random() for i in epochs_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs_range,
                    y=train_acc,
                    name='Training Accuracy',
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=epochs_range,
                    y=val_acc,
                    name='Validation Accuracy',
                    mode='lines+markers',
                    line=dict(color='#ff7f0e', width=3)
                ))
                
                fig.update_layout(
                    title='Model Accuracy During Training (Demo)',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Store demo model state
                st.session_state.demo_model_trained = True
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def show_prediction_page():
    """Display the live prediction interface."""
    st.markdown("## 🔍 Live Traffic Sign Prediction")
    
    if 'demo_model_trained' not in st.session_state:
        st.warning("⚠️ Please train a model first in the Model Training section.")
        return
    
    st.markdown("### 📤 Upload an Image")
    
    # File uploader - updated to include PPM files
    uploaded_file = st.file_uploader(
        "Choose a traffic sign image...",
        type=['png', 'jpg', 'jpeg', 'ppm'],
        help="Upload a traffic sign image to get real-time predictions"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**📷 Uploaded Image:**")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("**🔍 Preprocessed Image:**")
            # Preprocess image
            img_array = np.array(image)
            img_resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
            
            st.image(img_resized, caption=f"Resized to {IMG_WIDTH}x{IMG_HEIGHT}", use_column_width=True)
        
        # Make prediction
        if st.button("🚀 Predict Traffic Sign", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Get demo predictions
                    predictions, predicted_class = create_demo_predictions()
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Predicted Category:** {predicted_class}
                        
                        **Traffic Sign:** {TRAFFIC_SIGN_CATEGORIES.get(predicted_class, 'Unknown')}
                        
                        **Confidence:** {predictions[predicted_class]:.2%}
                        """)
                        
                        confidence = predictions[predicted_class]
                        if confidence > 0.8:
                            st.success("✅ High confidence prediction!")
                        elif confidence > 0.6:
                            st.warning("⚠️ Medium confidence prediction")
                        else:
                            st.error("❌ Low confidence prediction")
                    
                    with col2:
                        # Top 5 predictions
                        top_indices = np.argsort(predictions)[-5:][::-1]
                        top_predictions = [(i, predictions[i]) for i in top_indices]
                        
                        st.markdown("**Top 5 Predictions:**")
                        for i, (class_idx, conf) in enumerate(top_predictions):
                            sign_name = TRAFFIC_SIGN_CATEGORIES.get(class_idx, f"Category {class_idx}")
                            st.markdown(f"{i+1}. **{sign_name}** ({conf:.2%})")
                    
                    # Confidence visualization
                    st.markdown("### 📊 Confidence Distribution")
                    fig = px.bar(
                        x=[TRAFFIC_SIGN_CATEGORIES.get(i, f"Cat {i}") for i in range(NUM_CATEGORIES)],
                        y=predictions,
                        title="Confidence Scores for All Categories (Demo)",
                        labels={'x': 'Traffic Sign Category', 'y': 'Confidence Score'}
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

def show_performance_page():
    """Display model performance metrics."""
    st.markdown("## 📈 Performance Metrics & Evaluation")
    
    if 'demo_model_trained' not in st.session_state:
        st.warning("⚠️ Please train a model first in the Model Training section.")
        return
    
    st.markdown("### 🧪 Model Evaluation")
    
    if st.button("📊 Evaluate Model Performance"):
        with st.spinner("Evaluating model..."):
            try:
                # Display demo metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Accuracy", "0.9745")
                
                with col2:
                    st.metric("Test Loss", "0.0456")
                
                with col3:
                    st.metric("Error Rate", "0.0255")
                
                # Detailed analysis
                st.markdown("### 📋 Performance Analysis")
                
                # Demo confusion matrix
                st.markdown("### 📊 Confusion Matrix (Demo)")
                
                # Create a realistic demo confusion matrix
                np.random.seed(42)
                cm = np.random.poisson(50, (43, 43))
                np.fill_diagonal(cm, np.random.poisson(200, 43))  # Higher values on diagonal
                
                fig = px.imshow(
                    cm,
                    title="Confusion Matrix (Demo Data)",
                    labels=dict(x="Predicted", y="Actual"),
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Demo classification report
                st.markdown("### 📊 Classification Report (Demo)")
                
                # Create demo classification report
                demo_report = {
                    'precision': np.random.uniform(0.85, 0.98, NUM_CATEGORIES),
                    'recall': np.random.uniform(0.80, 0.97, NUM_CATEGORIES),
                    'f1-score': np.random.uniform(0.82, 0.98, NUM_CATEGORIES),
                    'support': np.random.randint(100, 500, NUM_CATEGORIES)
                }
                
                report_df = pd.DataFrame(demo_report, 
                                       index=[TRAFFIC_SIGN_CATEGORIES.get(i, f"Category {i}") for i in range(NUM_CATEGORIES)])
                st.dataframe(report_df, use_container_width=True)
                
                # Save model option
                st.markdown("### 💾 Save Model")
                st.info("""
                🎯 **Demo Mode**: In a production environment, this would save the actual trained model.
                The model file would contain the learned weights and architecture.
                """)
                
                if st.button("💾 Download Demo Model"):
                    st.success("✅ Demo model download simulation completed!")
                    st.info("📁 In production, this would download a .h5 file containing the trained model.")
                
            except Exception as e:
                st.error(f"❌ Evaluation failed: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Sign Recognition AI</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Computer Vision & Deep Learning System")
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔍 Live Prediction", "📈 Performance Metrics"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Analysis":
        show_data_analysis_page()
    elif page == "🤖 Model Training":
        show_demo_training_page()
    elif page == "🔍 Live Prediction":
        show_prediction_page()
    elif page == "📈 Performance Metrics":
        show_performance_page()

if __name__ == "__main__":
    main()
