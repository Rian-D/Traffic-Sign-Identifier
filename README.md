# 🚦 Traffic Sign Recognition AI - Web Application

A professional, interactive web application for traffic sign recognition using deep learning and computer vision. This project demonstrates advanced machine learning techniques with a beautiful, production-ready web interface.

<figure class="video_container">
  <iframe src="traffic_project_vid/test.mp4" frameborder="0" allowfullscreen="true"> 
</iframe>
</figure>

## 🌟 Features

### 🎯 **Core Functionality**
- **43 Traffic Sign Categories** from German Traffic Sign Recognition Benchmark (GTSRB)
- **Real-time Image Classification** with confidence scores
- **Interactive Training Interface** with live progress tracking
- **Comprehensive Analytics** and performance visualisation
- **Professional Web Interface** suitable for production deployment

### 🛠️ **Technical Capabilities**
- **Computer Vision**: Image processing and feature extraction using OpenCV
- **Deep Learning**: Convolutional Neural Networks (CNN) with TensorFlow
- **Data Science**: Statistical analysis and model evaluation with scikit-learn
- **Web Development**: Interactive Streamlit interface with Plotly visualisations

### 📊 **Interactive Sections**
1. **🏠 Home**: Project overview and system statistics
2. **📊 Data Analysis**: Dataset exploration and visualisation
3. **🤖 Model Training**: Interactive neural network training
4. **🔍 Live Prediction**: Real-time image classification
5. **📈 Performance Metrics**: Comprehensive model evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Conda environment manager
- Traffic sign dataset (GTSRB format)

### Installation

1. **Clone or download the project**
   ```bash
   cd traffic-2
   ```

2. **Activate the conda environment**
   ```bash
   conda activate traffic
   ```

3. **Run the web application**
   ```bash
   streamlit run traffic_app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in your terminal


**Supported Image Formats**: PNG, JPG, JPEG, PPM

## 🎮 How to Use

### 1. **Data Analysis** 📊
- Navigate to "📊 Data Analysis" in the sidebar
- Enter your dataset path (default: "gtsrb")
- Click "🔍 Analyze Dataset" to load and explore your data
- View category distribution, sample images, and statistics

### 2. **Model Training** 🤖
- After loading data, go to "🤖 Model Training"
- Adjust training parameters (epochs, batch size, test split)
- Click "🚀 Start Training" to begin neural network training
- Monitor real-time progress and view training results

### 3. **Live Prediction** 🔍
- Train a model first, then go to "🔍 Live Prediction"
- Upload any traffic sign image
- Get instant predictions with confidence scores
- View top 5 predictions and confidence distributions

### 4. **Performance Metrics** 📈
- Evaluate your trained model's performance
- View confusion matrix and classification reports
- Download the trained model for deployment


### Technologies Used
- **TensorFlow 2.x**: Deep learning framework
- **OpenCV**: Computer vision and image processing
- **Scikit-learn**: Machine learning utilities
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualisations
- **Pandas**: Data manipulation and analysis


## 📈 Future Enhancements

- **Real-time Video Processing**: Live camera feed analysis
- **Mobile App**: iOS/Android deployment
- **Cloud Deployment**: AWS/Azure integration
- **Advanced Architectures**: ResNet, EfficientNet, Vision Transformers
- **Multi-language Support**: International traffic sign recognition

Streamlit demo: what the UI could potentially look like:


## 📝 License

This project is open source and available under the MIT License.
