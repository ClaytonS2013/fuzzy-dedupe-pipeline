# 🔄 Fuzzy Dedupe Pipeline

An intelligent data deduplication system that syncs between Google Sheets and Supabase, with AI-powered fuzzy matching capabilities.

## ✨ Features

- **Fuzzy Matching**: Intelligent duplicate detection using multiple algorithms
- **AI Enhancement**: Claude AI integration for smart validation
- **Bidirectional Sync**: Seamless sync between Google Sheets and Supabase
- **Automated Processing**: Scheduled deduplication runs
- **Machine Learning**: Semantic similarity with sentence transformers
- **Production Ready**: Docker containerization for easy deployment

## 📋 Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- Supabase account and project
- Google Cloud service account with Sheets API access
- (Optional) Anthropic API key for Claude AI features

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/fuzzy-dedupe-pipeline.git
   cd fuzzy-dedupe-pipeline
```

2. **Set up environment variables**
```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   nano .env
```

3. **Run with Docker Compose**
```bash
   docker-compose up -d
```

4. **Check logs**
```bash
   docker logs -f fuzzy-dedupe-pipeline
```

### Option 2: Local Development

1. **Clone and setup**
```bash
   git clone https://github.com/yourusername/fuzzy-dedupe-pipeline.git
   cd fuzzy-dedupe-pipeline
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Configure environment**
```bash
   cp .env.example .env
   # Edit .env with your credentials
```

5. **Run the pipeline**
```bash
   python main.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:
```bash
# Supabase Configuration (Required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# Google Sheets Configuration (Required)
SPREADSHEET_ID=your-google-spreadsheet-id
GOOGLE_SHEETS_CREDS_PATH=/app/credentials/google-creds.json
# OR use JSON directly:
# GOOGLE_SHEETS_CREDS_JSON='{"type": "service_account", ...}'

# AI Enhancement (Optional)
ANTHROPIC_API_KEY=sk-ant-your-api-key

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
SCHEDULE_CRON=0 */6 * * *  # Run every 6 hours
BATCH_SIZE=100
DRY_RUN=false
```

### Google Sheets Setup

1. **Create a service account** in Google Cloud Console
2. **Enable Google Sheets API**
3. **Download credentials JSON**
4. **Share your spreadsheet** with the service account email
5. **Add the spreadsheet ID** to your `.env` file

### Supabase Setup

1. **Create a new project** at [supabase.com](https://supabase.com)
2. **Get your project URL and anon key** from Settings → API
3. **Create required tables** (schema provided in `/database/schema.sql`)
4. **Add credentials** to your `.env` file

## 🐳 Docker Commands

### Build & Run
```bash
# Build image
docker build -t fuzzy-dedupe-pipeline .

# Run container
docker run -d --name fuzzy-dedupe --env-file .env fuzzy-dedupe-pipeline

# Using Docker Compose
docker-compose up -d
```

### Management
```bash
# View logs
docker logs -f fuzzy-dedupe-pipeline

# Stop container
docker-compose down

# Enter container shell
docker exec -it fuzzy-dedupe-pipeline /bin/bash

# View resource usage
docker stats fuzzy-dedupe-pipeline

# Rebuild
docker-compose build --no-cache
```

## 📁 Project Structure
```
fuzzy-dedupe-pipeline/
├── dedupe_logic/
│   ├── __init__.py
│   └── processor.py      # Core deduplication logic
├── sheets_sync/
│   ├── __init__.py
│   └── sync.py          # Google Sheets integration
├── main.py               # Main application entry point
├── startup.sh            # Container startup script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── .env.example         # Environment variables template
└── README.md           # This file
```

## 🔧 How It Works

1. **Data Ingestion**: Fetches records from Google Sheets and Supabase
2. **Fuzzy Matching**: Applies multiple algorithms to identify duplicates:
   - Levenshtein distance
   - Jaro-Winkler similarity
   - Semantic similarity (using embeddings)
3. **AI Validation**: Optional Claude AI validation for edge cases
4. **Deduplication**: Merges or removes duplicate records
5. **Sync Back**: Updates both Google Sheets and Supabase
6. **Logging**: Comprehensive logging of all operations

## 📊 Monitoring

### Health Check
```bash
# Check container health
docker inspect fuzzy-dedupe-pipeline --format='{{.State.Health.Status}}'

# View metrics
docker stats fuzzy-dedupe-pipeline
```

### Logs
- Container logs: `./logs/app.log`
- Error logs: `./logs/error.log`
- Dedupe reports: `./logs/dedupe_report_YYYY-MM-DD.json`

## 🛠️ Troubleshooting

### Common Issues

1. **Docker build fails with "supabase==2.0.3 not found"**
   - Solution: Ensure `requirements.txt` uses `--extra-index-url` instead of `--index-url`

2. **"Cannot connect to Supabase"**
   - Check your SUPABASE_URL and SUPABASE_KEY in `.env`
   - Ensure your Supabase project is active

3. **"Google Sheets authentication failed"**
   - Verify service account credentials
   - Check if spreadsheet is shared with service account email

4. **Out of memory errors**
   - Increase Docker memory limit in `docker-compose.yml`
   - Reduce BATCH_SIZE in `.env`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Support

- Create an [Issue](https://github.com/yourusername/fuzzy-dedupe-pipeline/issues) for bugs
- Start a [Discussion](https://github.com/yourusername/fuzzy-dedupe-pipeline/discussions) for questions
- Check [Wiki](https://github.com/yourusername/fuzzy-dedupe-pipeline/wiki) for detailed guides

## 🙏 Acknowledgments

- [Supabase](https://supabase.com) for the backend infrastructure
- [Google Sheets API](https://developers.google.com/sheets/api) for spreadsheet integration
- [Anthropic Claude](https://anthropic.com) for AI capabilities
- [Sentence Transformers](https://www.sbert.net/) for semantic similarity

---

**Built with ❤️ for data quality**
```

---

## **Where to put this README.md file:**

1. **In your GitHub repository root directory** (same level as Dockerfile, requirements.txt, etc.)
```
   fuzzy-dedupe-pipeline/
   ├── README.md          <-- Put it here
   ├── Dockerfile
   ├── requirements.txt
   ├── docker-compose.yml
   ├── main.py
   └── ...other files
