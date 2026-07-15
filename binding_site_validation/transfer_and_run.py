#!/usr/bin/env python3
"""Transfer binding validation pipeline to server and execute."""
import paramiko
import os, sys, base64, time

HOST = os.environ.get('MMDCG_DTA_SSH_HOST')
PORT = int(os.environ.get('MMDCG_DTA_SSH_PORT', '22'))
USER = os.environ.get('MMDCG_DTA_SSH_USER')
PASS = os.environ.get('MMDCG_DTA_SSH_PASSWORD')
REMOTE_BASE = os.environ.get('MMDCG_DTA_REMOTE_BASE', '~/protein_ligand/MMDCG-DTA/MMDCG-DTA-main')
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

def ssh_connect():
    missing = [name for name, value in {
        'MMDCG_DTA_SSH_HOST': HOST,
        'MMDCG_DTA_SSH_USER': USER,
        'MMDCG_DTA_SSH_PASSWORD': PASS,
    }.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client

def ssh_exec(client, cmd, timeout=120):
    print(f"  [EXEC] {cmd[:100]}...")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err:
        print(f"  [STDERR] {err[:200]}")
    return out, err

def upload_file(client, local_path, remote_path):
    """Upload a file to the server."""
    print(f"  [UPLOAD] {os.path.basename(local_path)} -> {remote_path}")
    sftp = client.open_sftp()
    try:
        sftp.put(local_path, remote_path)
    finally:
        sftp.close()

def upload_dir_recursive(client, local_dir, remote_dir):
    """Upload all files in a directory recursively."""
    sftp = client.open_sftp()
    # Ensure remote dir exists
    try:
        sftp.mkdir(remote_dir)
    except:
        pass

    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = f"{remote_dir}/{item}"
        if os.path.isfile(local_path):
            print(f"  [UPLOAD] {item} -> {remote_path}")
            sftp.put(local_path, remote_path)
        elif os.path.isdir(local_path) and not item.startswith('.') and item not in ('__pycache__',):
            try:
                sftp.mkdir(remote_path)
            except:
                pass
            upload_dir_recursive(client, local_path, remote_path)

    sftp.close()

def main():
    print("=" * 60)
    print("MMDCG-DTA Binding Site Validation - Server Transfer & Run")
    print("=" * 60)

    # Connect
    print("\n[1/5] Connecting to server...")
    client = ssh_connect()
    print("  Connected!")

    # Create remote directories
    print("\n[2/5] Setting up remote directories...")
    remote_dir = f"{REMOTE_BASE}/binding_site_validation"
    ssh_exec(client, f"mkdir -p {remote_dir}/figures {remote_dir}/checkpoints {remote_dir}/pdbs {remote_dir}/results")

    # Upload files
    print("\n[3/5] Uploading files...")
    upload_dir_recursive(client, LOCAL_DIR, remote_dir)

    # Verify upload
    out, _ = ssh_exec(client, f"ls -la {remote_dir}/")
    print(f"  Remote files: {out}")

    # Run pipeline
    print("\n[4/5] Running pipeline on server...")
    conda_setup = "source /root/anaconda3/etc/profile.d/conda.sh && conda activate mmdcg_dta_env"
    full_cmd = (
        f"cd {REMOTE_BASE} && "
        f"{conda_setup} && "
        f"export PYTHONUNBUFFERED=1 && "
        f"export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:$LD_LIBRARY_PATH && "
        f"python -u binding_site_validation/binding_validation_pipeline.py "
        f"2>&1 | tee binding_site_validation/pipeline_run.log"
    )

    print(f"  Command: {full_cmd[:150]}...")
    print(f"  This may take 15-30 minutes...")

    # Use exec_command with a longer timeout
    try:
        stdin, stdout, stderr = client.exec_command(full_cmd, timeout=1800)  # 30 min timeout
        # Stream output
        while True:
            line = stdout.readline()
            if not line:
                break
            print(f"  [SERVER] {line.rstrip()}")
        err_out = stderr.read().decode()
        if err_out:
            print(f"  [ERR] {err_out[:500]}")
    except Exception as e:
        print(f"  [WARN] Pipeline may still be running: {e}")

    # Check results
    print("\n[5/5] Checking results...")
    time.sleep(3)

    out, _ = ssh_exec(client, f"ls -la {remote_dir}/figures/ 2>/dev/null")
    print(f"  Figures: {out}")

    out, _ = ssh_exec(client, f"ls -la {remote_dir}/results/ 2>/dev/null")
    print(f"  Results: {out}")

    # Also do analysis-only mode if main pipeline completed
    print("\n[Extra] Running analysis-only mode for supplementary figures...")
    full_cmd2 = (
        f"cd {REMOTE_BASE} && "
        f"{conda_setup} && "
        f"export PYTHONUNBUFFERED=1 && "
        f"export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:$LD_LIBRARY_PATH && "
        f"python -u binding_site_validation/binding_validation_pipeline.py --analysis-only "
        f"2>&1 | tee -a binding_site_validation/pipeline_run.log"
    )
    try:
        stdin, stdout, stderr = client.exec_command(full_cmd2, timeout=600)
        while True:
            line = stdout.readline()
            if not line:
                break
            print(f"  [ANALYSIS] {line.rstrip()}")
    except Exception as e:
        print(f"  [WARN] Analysis may still be running: {e}")

    # Generate report
    print("\n[Report] Generating validation report...")
    out, _ = ssh_exec(client, (
        f"cd {REMOTE_BASE} && {conda_setup} && "
        f"python -u binding_site_validation/generate_report.py 2>&1"
    ))
    print(f"  {out}")

    # Download results back to local
    print("\n[Download] Fetching results...")
    sftp = client.open_sftp()
    local_fig_dir = os.path.join(LOCAL_DIR, 'figures')
    os.makedirs(local_fig_dir, exist_ok=True)

    try:
        remote_figs = sftp.listdir(f"{remote_dir}/figures")
        for f in remote_figs:
            if f.endswith('.png'):
                remote_path = f"{remote_dir}/figures/{f}"
                local_path = os.path.join(local_fig_dir, f)
                sftp.get(remote_path, local_path)
                print(f"  Downloaded: {f}")

        # Download report
        report_remote = f"{remote_dir}/MMDCG-DTA_结合位点验证报告.md"
        report_local = os.path.join(LOCAL_DIR, 'MMDCG-DTA_结合位点验证报告.md')
        try:
            sftp.get(report_remote, report_local)
            print(f"  Downloaded: MMDCG-DTA_结合位点验证报告.md")
        except:
            print(f"  [WARN] Report not found on server")

        # Download results JSON
        for res_file in ['binding_validation_results.json', 'per_residue_energies.pkl']:
            remote_rf = f"{remote_dir}/results/{res_file}"
            local_rf = os.path.join(LOCAL_DIR, 'results', res_file)
            try:
                sftp.get(remote_rf, local_rf)
                print(f"  Downloaded: {res_file}")
            except:
                print(f"  [WARN] {res_file} not found")

    finally:
        sftp.close()

    client.close()
    print("\n" + "=" * 60)
    print("Transfer & Run Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
