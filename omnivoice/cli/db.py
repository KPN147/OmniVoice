import os
import shutil
import datetime
import uuid
import pandas as pd

class OmniVoiceDB:
    def __init__(self, base_dir=None):
    #     # Determine the base directory
    #     try:
    #         IN_COLAB = True
    #         from google.colab import drive
    #         drive.mount('/content/drive')
    #         mount_drive = True
    #     except:
    #         mount_drive = False
    #     if mount_drive:
    #         self.base_dir = "/content/drive/MyDrive/OminiVoiceDB"
    #     else:
    #         self.base_dir = "Omini_Voice_DB"
        if os.path.exists('/content/drive/MyDrive'):
                self.base_dir = "/content/drive/MyDrive/OminiVoiceDB"
                print("Google Drive found! Saving to Google Drive.")
        else:
            self.base_dir = "Omini_Voice_DB"
            print("Google Drive NOT found. Saving locally.")

        self.audio_dir = os.path.join(self.base_dir, "audios")
        self.clones_csv = os.path.join(self.base_dir, "VoiceClones.csv")
        self.history_csv = os.path.join(self.base_dir, "History.csv")
        self.initialized = False

        try:
            self.setup_db()
            self.initialized = True
            print(f"Database initialized at {self.base_dir}")
        except Exception as e:
            print(f"Warning: Failed to initialize DB: {e}")

    def setup_db(self):
        os.makedirs(self.audio_dir, exist_ok=True)
        
        if not os.path.exists(self.clones_csv):
            df = pd.DataFrame(columns=["ID", "Name", "Ref Audio Drive ID", "Ref Text", "Created At"])
            df.to_csv(self.clones_csv, index=False)
            
        if not os.path.exists(self.history_csv):
            df = pd.DataFrame(columns=["ID", "Text", "Voice Clone Used", "Status", "Output Audio Drive ID", "Created At"])
            df.to_csv(self.history_csv, index=False)

    # === File Operations ===
    def upload_audio(self, file_path, name):
        """Copy an audio file to the Omni Voice folder and return the new path."""
        if not file_path or not os.path.exists(file_path):
            return None
        
        ext = os.path.splitext(file_path)[1]
        # Keep original extension, use name or generate uuid
        if not name.endswith(ext) and not name.endswith('.wav') and not name.endswith('.mp3'):
            name += ext
        
        dest_path = os.path.join(self.audio_dir, name)
        shutil.copy2(file_path, dest_path)
        return dest_path
    
    def download_audio(self, source_path, output_path):
        """Copy an audio file by path to output_path."""
        if not source_path or not os.path.exists(source_path):
            return False
        try:
            shutil.copy2(source_path, output_path)
            return True
        except Exception:
            return False

    # === Voice Clones ===
    def add_voice_clone(self, name, audio_path, ref_text):
        audio_id = self.upload_audio(audio_path, f"{name}_ref_{uuid.uuid4().hex[:8]}.wav")
        clone_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {
            "ID": clone_id,
            "Name": name,
            "Ref Audio Drive ID": audio_id,
            "Ref Text": ref_text,
            "Created At": created_at
        }
        
        df = pd.read_csv(self.clones_csv)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.clones_csv, index=False)
        return clone_id

    def get_voice_clones(self):
        if not os.path.exists(self.clones_csv):
            return pd.DataFrame(columns=["ID", "Name", "Ref Audio Drive ID", "Ref Text", "Created At"])
        return pd.read_csv(self.clones_csv).fillna("")
    
    def delete_voice_clone(self, clone_id):
        if not os.path.exists(self.clones_csv):
            return False
        df = pd.read_csv(self.clones_csv)
        clone_id = str(clone_id)
        
        mask = df["ID"].astype(str) == clone_id
        if not mask.any():
            return False
            
        # Delete audio files
        for audio_path in df[mask]["Ref Audio Drive ID"]:
            if pd.notna(audio_path) and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
                    
        df = df[~mask]
        df.to_csv(self.clones_csv, index=False)
        return True

    # === History ===
    def add_history(self, text, clone_name, status, audio_path=None):
        audio_id = ""
        if audio_path and status == "Success":
            audio_id = self.upload_audio(audio_path, f"generated_{uuid.uuid4().hex[:8]}.mp3")
            
        history_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = {
            "ID": history_id,
            "Text": text,
            "Voice Clone Used": clone_name,
            "Status": status,
            "Output Audio Drive ID": audio_id,
            "Created At": created_at
        }
        
        df = pd.read_csv(self.history_csv)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.history_csv, index=False)
        return history_id

    def get_history(self):
        if not os.path.exists(self.history_csv):
            return pd.DataFrame(columns=["ID", "Text", "Voice Clone Used", "Status", "Output Audio Drive ID", "Created At"])
        return pd.read_csv(self.history_csv).fillna("")

    def update_history_status(self, history_id, status, audio_path=None):
        if not os.path.exists(self.history_csv):
            return False
        df = pd.read_csv(self.history_csv)
        history_id = str(history_id)
        
        mask = df["ID"].astype(str) == history_id
        if not mask.any():
            return False
            
        idx = df[mask].index[0]
        df.at[idx, "Status"] = status
        
        if audio_path and status == "Success":
            audio_id = self.upload_audio(audio_path, f"generated_{uuid.uuid4().hex[:8]}.mp3")
            df.at[idx, "Output Audio Drive ID"] = audio_id
            
        df.to_csv(self.history_csv, index=False)
        return True

    def delete_history(self, history_id):
        if not os.path.exists(self.history_csv):
            return False
        df = pd.read_csv(self.history_csv)
        history_id = str(history_id)
        
        mask = df["ID"].astype(str) == history_id
        if not mask.any():
            return False
            
        # Delete audio files
        for audio_path in df[mask]["Output Audio Drive ID"]:
            if pd.notna(audio_path) and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
                    
        df = df[~mask]
        df.to_csv(self.history_csv, index=False)
        return True
