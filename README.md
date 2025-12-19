# 🩺 Medical Assistant Chatbot

A sophisticated RAG (Retrieval-Augmented Generation) based medical chatbot powered by fine-tuned LLaMA 3 and advanced NLP techniques. This system combines document retrieval, semantic search, and large language models to provide accurate medical information and assistance.

## Demo Video 
https://github.com/MerayBasanti/Medical-Assistant/blob/main/demo.mp4

## 🌟 Features

- **Fine-Tuned LLaMA 3 Model**: Custom-trained on 112K+ medical Q&A pairs for domain-specific accuracy
- **RAG Architecture**: Retrieval-Augmented Generation for context-aware responses
- **Vector Database**: ChromaDB integration for efficient semantic search
- **Modern Web Interface**: Clean, responsive UI with dark mode support
- **GGUF Optimization**: Quantized model (Q4_K_M) for efficient inference
- **Real-time Chat**: FastAPI backend with async processing

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset & Fine-Tuning](#dataset--fine-tuning)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│    RAG Pipeline             │
│  ┌─────────────────────┐   │
│  │ Document Retrieval  │   │
│  │   (ChromaDB)        │   │
│  └─────────┬───────────┘   │
│            │                │
│            ▼                │
│  ┌─────────────────────┐   │
│  │  Context Injection  │   │
│  └─────────┬───────────┘   │
│            │                │
│            ▼                │
│  ┌─────────────────────┐   │
│  │  LLaMA 3 (GGUF)     │   │
│  │  Fine-tuned Model   │   │
│  └─────────┬───────────┘   │
└────────────┼───────────────┘
             │
             ▼
      ┌─────────────┐
      │  Response   │
      └─────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- Ollama (for model serving)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

### Step 5: Pull the Fine-Tuned Model

```bash
ollama pull hf.co/sathvik123/llama3-ChatDoc
```

### Step 6: Set Up Vector Database

The vector database will be automatically initialized on first run. Ensure your medical documents are placed in the appropriate directory.

## 📊 Dataset & Fine-Tuning

### Dataset Preparation

The model was trained on the **ChatDoctor HealthcareMagic** dataset with 112,165 medical Q&A pairs.

**Dataset Statistics:**
- Training samples: 89,732
- Test samples: 22,433
- Source: `lavita/medical-qa-datasets`

**Data Processing Pipeline:**
1. Load raw medical Q&A data
2. Format with LLaMA 3 chat template
3. Split into train/test (80/20)
4. Push to HuggingFace Hub

```python
# Example prompt format
<|start_header_id|>system<|end_header_id|>
If you are a doctor, please answer the medical questions based on the patient's description.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{answer}
<|eot_id|>
```

### Fine-Tuning Process

**Model:** `unsloth/llama-3-8b-Instruct-bnb-4bit`

**Training Configuration:**
- **LoRA Parameters:**
  - Rank (r): 16
  - Alpha: 16
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0

- **Training Arguments:**
  - Batch size: 2 (per device)
  - Gradient accumulation: 4 steps
  - Effective batch size: 8
  - Learning rate: 2e-4
  - Optimizer: AdamW 8-bit
  - Max steps: 100
  - Warmup steps: 5
  - Weight decay: 0.01

**Quantization:**
- Format: GGUF Q4_K_M
- Model size: ~4.7GB (compressed from 16GB)
- Inference speed: 2x faster than full precision

### Training Results

The model was trained using the Unsloth framework with 4-bit quantization:

```
Training Loss Progression:
Initial: 2.30
Final: 2.21
Training Time: ~15 minutes (100 steps)
```

## 📈 Model Performance

Evaluation on 100 test samples:

| Metric | Score |
|--------|-------|
| **ROUGE-1** | 0.2929 |
| **ROUGE-2** | 0.0397 |
| **ROUGE-L** | 0.1472 |
| **METEOR** | 0.2438 |

**Evaluation Code:**
```python
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
```

The model shows strong performance in medical domain tasks with good semantic understanding while maintaining computational efficiency.

## 💻 Usage

### Starting the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Accessing the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### Example Queries

```
User: "What are the treatments for vitamin D-dependent rickets?"

Bot: "Vitamin D-dependent rickets can be treated through:
1. Vitamin D supplements (high doses)
2. Calcitriol supplements
3. Calcium supplementation
4. Regular monitoring..."
```

### Programmatic Usage

```python
from RAG.rag import RAG_chatbot

bot = RAG_chatbot()
response = await bot.get_response("What causes diabetes?")
print(response)
```

## 📡 API Documentation

### POST /chat

Send a medical query and receive an AI-generated response.

**Request:**
```json
{
  "user_query": "What are the symptoms of flu?"
}
```

**Response:**
```json
{
  "answer": "The common symptoms of flu include..."
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_query": "What causes headaches?"}'
```

## 📁 Project Structure

```
medical-chatbot/
├── Fine-Tuning/
│   ├── Fine_tuning.ipynb          # Model training pipeline
│   ├── dataset_prep.ipynb         # Data preprocessing
│   └── evaluation.ipynb           # Model evaluation
├── RAG/
│   └── rag.py                     # RAG implementation
├── templates/
│   └── index.html                 # Web interface
├── app.py                         # FastAPI application
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🛠️ Technologies Used

### Core Framework
- **FastAPI**: High-performance web framework
- **LangChain**: LLM orchestration and RAG pipeline
- **Ollama**: Local LLM serving

### Machine Learning
- **Unsloth**: Efficient fine-tuning framework
- **Transformers**: HuggingFace model loading
- **PEFT/LoRA**: Parameter-efficient fine-tuning
- **BitsAndBytes**: 4-bit quantization

### Vector Database
- **ChromaDB**: Vector storage and semantic search
- **Sentence-Transformers**: Embedding generation

### Evaluation
- **Evaluate**: Metrics computation
- **ROUGE**: Text similarity scoring
- **METEOR**: Machine translation evaluation

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Vanilla JavaScript**: Real-time chat functionality

## 🎯 Key Features Explained

### 1. RAG Pipeline
The system retrieves relevant medical documents from ChromaDB and injects them as context into the LLM prompt, ensuring responses are grounded in reliable information.

### 2. Fine-Tuned Model
LLaMA 3 8B was fine-tuned on medical Q&A data using LoRA, adapting it specifically for medical conversations while maintaining general language capabilities.

### 3. Efficient Inference
GGUF quantization reduces model size by 70% while maintaining response quality, enabling deployment on consumer hardware.

### 4. Dark Mode UI
Modern, accessible interface with theme persistence using localStorage.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This chatbot is for **informational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with questions about medical conditions.

## 🙏 Acknowledgments

- **LLaMA 3** by Meta AI
- **Unsloth** for efficient fine-tuning
- **ChatDoctor** dataset by HealthcareMagic
- **Ollama** for local model serving
- **LangChain** for RAG framework

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact via HuggingFace discussions

---

**Star ⭐ this repository if you find it helpful!**
