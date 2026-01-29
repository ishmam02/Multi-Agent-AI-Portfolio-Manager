# Google Colab Setup Guide

## Step 1: Sign Up for Google Colab Pro (Optional but Recommended)

Google Colab Pro gives you access to better GPUs and longer runtime sessions. If you have an education email, you can get special pricing.

**What to do:**

- Visit [Google Colab Signup](https://colab.research.google.com/signup)
- Click "Sign up for Colab Pro"
- Use your education email to verify and get the pro for free

**Benefits of Colab Pro:**

- Access to premium GPUs (H100, A100)
- Longer session timeouts (12 hours vs 30 minutes on free)
- Higher memory allocation
- Priority access during peak hours

> **Note:** If you don't have Colab Pro, you can still use the free version with regular GPUs like T4.

---

## Step 2: Create a New Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **"+ New notebook"** in the bottom left
3. Give your notebook a name
4. You're now ready to start coding!

---

## Step 3: Configure GPU Runtime

GPU acceleration is essential for training deep learning models. Here's how to enable it:

### Steps to Enable GPU:

1. Click **"Runtime"** in the top menu
2. Select **"Change runtime type"**
3. A dialog box will appear

### Hardware Accelerator Options:

| Hardware     | VRAM | Availability | Speed     | Best For                                          |
| ------------ | ---- | ------------ | --------- | ------------------------------------------------- |
| **CPU**      | 12GB | Free         | Slowest   | Development, testing                              |
| **T4 GPU**   | 16GB | Free         | Fast      | Basic training, lightweight models                |
| **L4 GPU**   | 24GB | Pro          | Faster    | Medium-sized models                               |
| **A100 GPU** | 40GB | Pro          | Very Fast | Large models, heavy training                      |
| **H100 GPU** | 80GB | Pro          | Fastest   | Very large models, long training                  |
| **TPU v5e**  | 8GB  | Pro          | Very Fast | Suitable for specific workloads (JAX, TensorFlow) |
| **TPU v6e**  | 8GB  | Pro          | Fastest   | Latest TPU, experimental                          |

**Recommended Selection:**

- **For beginners/testing code:** Select **T4 GPU** (free, sufficient for most projects)
- **For serious training:** Select **A100 GPU** or **H100 GPU** with Colab Pro
- **For experiments:** L4 GPU is a good middle ground with Pro
- **For specific ML frameworks:** TPU options (v5e, v6e) if your code supports them

### How to Select:

1. In the runtime dialog, find **"Hardware accelerator"**
2. Select **"GPU"** (or "TPU" if you prefer)
3. Click **"Save"**

> The notebook will restart with your new runtime configuration.

---

## Step 4: Clone Your Git Repository

Colab doesn't persistently save files uploaded through the interface. Instead, you should clone your GitHub repository to work with your code.

### Why Clone Instead of Upload?

- ✅ Your changes are automatically versioned
- ✅ Easy to sync across sessions
- ✅ Can push updates back to GitHub
- ✅ Persistent access to your project

### Steps to Clone:

1. **Get your repository URL:**
   - Go to your GitHub repository
   - Click **"Code"** (green button)
   - Copy the HTTPS URL

2. **In your Colab notebook, create a new cell and run:**

   ```bash
   !git clone _link_
   ```

3. **Navigate to the project directory:**
   ```bash
   %cd project_directory
   ```

After cloning, your project files will be available in Colab's `/content/` directory.

---

## Step 5: Managing Code Changes

### Option A: Make Changes in Colab + Push to GitHub

If you're making small changes in Colab:

1. **Configure git (first time only):**

   ```bash
   !git config --global user.name "Your Name"
   !git config --global user.email "your.email@example.com"
   ```

2. **Make your code changes** in the cloned repository

3. **Stage and commit changes:**

   ```bash
   !git add .
   !git commit -m "Your commit message"
   !git push origin main
   ```

4. **Authenticate with GitHub:**
   - You may need to use a Personal Access Token instead of your password
   - [Create a PAT here](https://github.com/settings/tokens)

### Option B: Make Changes in Your IDE + Pull in Colab

This is often easier for major refactoring:

1. **Make changes** in your favorite IDE (VSCode, PyCharm, etc.)
2. **Push to GitHub:**

   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

3. **In Colab, pull the latest changes:**
   ```bash
   !cd Multi-Agent-AI-Portfolio-Manager && git pull origin main
   ```

---

## Useful Colab Commands

| Command                     | Purpose                         |
| --------------------------- | ------------------------------- |
| `!nvidia-smi`               | Check GPU usage and memory      |
| `!df -h`                    | Check disk space                |
| `!pip install package_name` | Install Python packages         |
| `%cd /path`                 | Change directory                |
| `!ls`                       | List files in current directory |

---

## Troubleshooting

### Issue: "CUDA out of memory"

- Reduce batch size in your training parameters
- Use a smaller model
- Request a better GPU runtime

### Issue: "Permission denied" when pushing to GitHub

- Use a Personal Access Token instead of your password
- Or set up SSH keys (more advanced)

### Issue: "Session crashed"

- This happens after ~12 hours (free) or longer on Pro
- Save your progress frequently with `!git push`
- Re-run the clone and setup commands in a new session

---

## Tips for Success

1. **Always push your work:** Don't rely on Colab's storage
2. **Use checkpoints:** Save model weights frequently
3. **Monitor resources:** Use `nvidia-smi` to watch GPU usage
4. **Start small:** Test with small datasets before full training
