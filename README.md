# 🧠 Bayes MCP - Interactive Bayesian Analysis Suite

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Interface-Jupyter-orange)](https://jupyter.org)
[![Bayesian](https://img.shields.io/badge/Method-Bayesian-green)](https://en.wikipedia.org/wiki/Bayesian_inference)

**Interactive Bayesian analysis toolkit** with hands-on notebooks for business decisions, medical diagnosis, and financial risk assessment. Learn and apply Bayesian methods through real-world scenarios.

## 🎯 What's Inside

### 📊 **Interactive Notebooks** (Start Here!)
- **[Overview & Fundamentals](notebooks/00_bayesian_analysis_overview.ipynb)** - Learn Bayesian concepts interactively
- **[A/B Testing Analysis](notebooks/01_ab_testing_interactive.ipynb)** - Business decision optimization
- **[Medical Diagnosis](notebooks/02_medical_diagnosis_interactive.ipynb)** - Evidence-based clinical decisions  
- **[Financial Risk Assessment](notebooks/03_financial_risk_interactive.ipynb)** - Portfolio management & VaR

### 🛠️ **Core Engine**
- **Bayesian MCP Server** - Production-ready statistical engine
- **MCMC Sampling** - Advanced computational methods
- **Real-time Analysis** - Interactive parameter exploration
- **Visualization Tools** - Uncertainty-aware plotting
- **Logfire Integration** - Observability and monitoring for Bayesian computations

## 🚀 Quick Start

### 1. Setup Environment
```bash
git clone <repository-url>
cd bayes
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
pip install jupyter jupyterlab scipy matplotlib seaborn ipywidgets
```

### 2. Configure Logfire (Optional)
```bash
cp .env.example .env
# Edit .env and add your Logfire token for observability
```

### 3. Launch Interactive Notebooks
```bash
cd notebooks
jupyter lab
# Open 00_bayesian_analysis_overview.ipynb to start
```

### 4. Start with Fundamentals
- **Coin Flip Demo** - See Bayesian updating in action
- **Choose Your Domain** - A/B testing, medical, or financial
- **Interactive Exploration** - Adjust parameters, see real-time results

## 🎯 Use Cases & Applications

### 🧪 **A/B Testing Excellence**
- **Direct Probability**: Get P(B > A) instead of p-values
- **Expected Loss**: Quantify risk of wrong decisions
- **Revenue Impact**: Calculate business value of changes
- **Early Stopping**: Stop tests when confident enough

### 🏥 **Medical Diagnosis Support** 
- **Prior Integration**: Use prevalence and risk factors
- **Sequential Testing**: Optimal diagnostic strategies
- **Cost Analysis**: Balance test costs vs. information gain
- **Base Rate Awareness**: Avoid common diagnostic errors

### 💰 **Financial Risk Management**
- **Portfolio Optimization**: Risk-adjusted returns
- **Stress Testing**: Scenario-based risk assessment
- **VaR Calculation**: Regulatory compliance metrics
- **Market Regime**: Dynamic risk model adjustment

## 🧮 Why Bayesian Methods?

### **Traditional Approach Problems:**
- ❌ P-values don't answer business questions
- ❌ No uncertainty quantification
- ❌ Can't incorporate prior knowledge
- ❌ Fixed sample size requirements

### **Bayesian Solutions:**
- ✅ Direct probability statements: "95% chance B is better"
- ✅ Full uncertainty distributions
- ✅ Prior knowledge integration
- ✅ Sequential analysis with optimal stopping

## 📚 Documentation

### **Complete Guides** → [docs/](docs/)
- [Project Overview](docs/OVERVIEW.md) - Vision and architecture
- [Setup Guide](docs/SETUP_GUIDE.md) - Detailed installation
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) - Technical details
- [Contributing Guide](docs/CONTRIBUTING.md) - Development workflow

### **API Reference**
- **MCP Server**: Start with `python bayes_mcp.py --port 8002`
- **Engine Direct**: `from bayes_mcp.bayesian_engine import BayesianEngine`
- **REST API**: Full HTTP interface for integration

## 🎓 Learning Path

### **Beginner** 🌱
1. **[Bayesian Overview](notebooks/00_bayesian_analysis_overview.ipynb)** - Start here
2. **Coin flip demo** - Hands-on Bayes' theorem
3. **A/B testing basics** - Business applications
4. **Parameter exploration** - See how changes affect results

### **Intermediate** 🌿
1. **Medical diagnosis** - Evidence combination
2. **Financial risk** - Portfolio optimization  
3. **Compare approaches** - Bayesian vs. traditional
4. **Custom scenarios** - Adapt to your data

### **Advanced** 🌳
1. **Engine integration** - Production applications
2. **Model extension** - Add complexity
3. **API development** - Build services
4. **Contribution** - Enhance the platform

## 🎪 Live Demonstrations

### **Run Comprehensive Demos**
```bash
# All domains demonstration
python demos/master_demo.py

# Individual demos
python demos/ab_testing_demo.py      # Business optimization
python demos/medical_diagnosis_demo.py # Clinical decisions
python demos/financial_risk_demo.py   # Portfolio management
```

## 🎯 Success Metrics

### **Educational Impact**
- **Interactive Learning**: Hands-on Bayesian education
- **Real Scenarios**: Practical business applications  
- **Visual Feedback**: Immediate understanding
- **Progressive Complexity**: Learn at your pace

### **Business Value**
- **Better Decisions**: Uncertainty-aware choices
- **Risk Management**: Quantified downside protection
- **Cost Optimization**: Efficient resource allocation
- **Competitive Advantage**: Advanced analytical capabilities

## 📊 Observability with Logfire

When configured with a Logfire token, the Bayes MCP server provides comprehensive observability:

### **Performance Metrics**
- **MCMC Sampling Time** - Track computational performance
- **Model Creation** - Monitor model initialization
- **Variable Counts** - Track model complexity
- **API Response Times** - Service health monitoring

### **Tracing Features**
- **Request Tracing** - Full request lifecycle visibility
- **Span Tracking** - Detailed operation timing
- **Error Tracking** - Exception monitoring with context
- **Structured Logging** - Rich contextual information

### **Integration Benefits**
- **Performance Optimization** - Identify bottlenecks
- **Debugging Support** - Trace complex computations
- **Usage Analytics** - Understand model patterns
- **Production Monitoring** - Ensure reliability

## 🔧 Technical Stack

- **Python 3.8+** - Core language
- **Jupyter Lab** - Interactive interface
- **PyMC** - Bayesian computation engine
- **NumPy/SciPy** - Numerical computing
- **Matplotlib/Seaborn** - Visualization
- **IPyWidgets** - Interactive controls
- **FastAPI** - Production server (optional)

## 🏃‍♂️ Running Tests

```bash
# Full test suite (36 tests)
python -m pytest tests/ -v

# Quick functionality check
python -c "from bayes_mcp.bayesian_engine import BayesianEngine; print('✅ Engine ready!')"
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Test** your changes: `python -m pytest`
4. **Commit** changes: `git commit -m 'Add amazing feature'`
5. **Push** branch: `git push origin feature/amazing-feature`
6. **Open** Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🎉 Get Started Today!

**Ready to explore Bayesian analysis?**

```bash
# Launch the interactive suite
cd bayes/notebooks
jupyter lab
# Open 00_bayesian_analysis_overview.ipynb
```

**Transform your decision-making with principled uncertainty quantification!** 🚀

---

*Built for practitioners who need reliable, interpretable, and actionable statistical analysis.*