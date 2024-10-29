# Running VS Code Remotely on Lambda Labs (Mac-Friendly Guide)

Here’s how to set up VS Code to run files remotely on a Lambda Labs GPU instance. This guide includes steps for seamless integration, uploading files, and keeping your environment consistent.

### Step 1: Create and Set Up a Lambda Labs GPU Instance
1. Follow the [Getting Started with Lambda Cloud GPU Instances](https://lambdalabs.com/blog/getting-started-with-lambda-cloud-gpu-instances) tutorial to create your instance and configure SSH access.
2. Download your SSH key file (`FirstnameLastname.pem`) as instructed in the tutorial. For Mac users, place it in the `~/.ssh` directory and update permissions:
   ```bash
   mv ~/Downloads/FirstnameLastname.pem ~/.ssh/
   chmod 400 ~/.ssh/FirstnameLastname.pem
   ```

### Step 2: Install Remote-SSH Extension in VS Code
1. Open **VS Code**.
2. Go to the **Extensions** view (press `Cmd+Shift+X` on Mac).
3. Search for **Remote - SSH** and install it to enable remote connections from within VS Code.

### Step 3: Connect to the Lambda Instance
1. Open a terminal in VS Code.
2. Use the following SSH command to connect, replacing `FirstnameLastname.pem` with your key and `instance-ip-address` with your Lambda instance IP:
   ```bash
   ssh -i ~/.ssh/FirstnameLastname.pem ubuntu@instance-ip-address
   ```
3. Once connected, VS Code will prompt you to open folders on your Lambda instance, giving you access to your project files directly.

### Step 4: Upload Files to the Lambda Instance
There are two main ways to upload files to the instance:
1. **Drag and Drop**: In the VS Code **Explorer** panel, you can drag files from your Mac into any open folder on the remote Lambda instance. For example, drag in a Jupyter notebook to start using it remotely.
2. **Git Workflow**: Alternatively, push your code to a Git repository (e.g., GitHub), and then clone it each time you connect:
   ```bash
   git clone https://github.com/your-repo.git
   ```

### Step 5: Manage Packages Efficiently in Jupyter
To avoid re-installing packages every session, add installation commands as cells in your Jupyter notebook:
   ```python
   !pip install numpy pandas matplotlib
   ```
This way, your environment stays consistent, and you save time by managing dependencies directly within the notebook.

---

Now you're set up to work remotely on Lambda Labs with VS Code, right from your Mac. Happy coding!
