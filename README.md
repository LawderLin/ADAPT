# ADAPT: Automatic Development of Assessment and Psychological Tests

**ADAPT** is an LLM-based Multi-Agent system designed to automate the development of psychological assessment items. It leverages a team of specialized AI agents to generate, review, and refine test items based on psychometric theories.

This project is a local implementation inspired by the **LM-AIG framework** proposed by Lee et al. (2025).

## 🚀 Features

* **Multi-Agent Workflow:** Orchestrates specialized agents (Item Writer, Content Reviewer, Linguistic Reviewer, Bias Reviewer) to ensure high-quality item generation.
* **Local Privacy:** Built to run with local LLMs (via Ollama) to prevent sensitive test construct leakage.
* **Psychometric Validation:** Automated checks for readability, bias, and construct relevance.
* **Structured Output:** Utilizes JSON-enforced outputs for consistent downstream data analysis.

## 🛠️ Architecture

ADAPT follows the **LM-AIG** pipeline:
1.  **Input:** User provides a construct definition (e.g., "AI Anxiety").
2.  **Web Research:** Agents search for theoretical context and definition distinctiveness.
3.  **Generation:** The **Item Writer** creates an initial pool of items.
4.  **Review Loop:**
    * **Content Reviewer:** Checks if items match the construct definition.
    * **Linguistic Reviewer:** Checks for grammar and reading level (e.g., 7th-grade level).
    * **Bias Reviewer:** Audits for cultural, gender, or demographic bias.
5.  **Refinement:** The **Meta Editor** synthesizes feedback and finalizes the scale.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LawderLin/ADAPT.git
    cd ADAPT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama:**
    Ensure [Ollama](https://ollama.com/) is installed and running (`ollama serve`). Pull the model:
    ```bash
    ollama pull qwen3:8b
    ```

## 💻 Usage

1.  **Configure Agents:**
    Modify `config.py` to select your local model or adjust agent prompts.

2.  **Run the Workflow:**
    ```bash
    python main.py
    ```
    *Follow the CLI prompts to input your test construct.*

3.  **View Results:**
    Generated items and revision logs will be saved to the `lm_aig_workflow_results_XXXXXXXX_XXXXXX.txt`.

## ⚠️ Known Issues / TODO

- [ ] **Search Agent:** Integration with local search tools is currently experimental.
- [ ] **UI:** A web interface (Gradio/Streamlit) is in development.
- [ ] **Data Analysis:** Automated calculation of Content Validity Indices (CVI) is pending.

## 📄 Reference

This project implements concepts from:
> Lee, P., Son, M., & Jia, Z. (2025). *AI-powered Automatic Item Generation for Psychological Tests: A Conceptual Framework for an LLM-based Multi-Agent AIG System*. Journal of Business and Psychology. https://doi.org/10.1007/s10869-025-10067-y
