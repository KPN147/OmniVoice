import os
import io
import datetime
import uuid
import gspread
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

FOLDER_NAME = "Omini Voice"
SHEET_NAME = "OmniVoice_DB"

class OmniVoiceDB:
    def __init__(self, creds_path="credentials.json", token_path="token.json"):
        self.creds_path = creds_path
        self.token_path = token_path
        self.creds = None
        self.drive_service = None
        self.gc = None
        self.folder_id = None
        self.spreadsheet = None
        self.initialized = False
        
        # Attempt to connect
        try:
            self.authenticate()
            self.setup_drive()
            self.setup_sheets()
            self.initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize Google Drive DB: {e}")
            print(f"Please ensure {self.creds_path} exists and is valid.")

    def authenticate(self):
        if os.path.exists(self.token_path):
            self.creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.creds_path):
                    raise FileNotFoundError(f"Missing {self.creds_path}. Please download it from Google Cloud Console.")
                flow = InstalledAppFlow.from_client_secrets_file(self.creds_path, SCOPES)
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(self.creds.to_json())
        
        self.drive_service = build('drive', 'v3', credentials=self.creds)
        self.gc = gspread.authorize(self.creds)

    def setup_drive(self):
        # check if folder exists
        query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        
        if not items:
            # Create folder
            file_metadata = {
                'name': FOLDER_NAME,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = self.drive_service.files().create(body=file_metadata, fields='id').execute()
            self.folder_id = folder.get('id')
            print(f"Created folder '{FOLDER_NAME}' with ID: {self.folder_id}")
        else:
            self.folder_id = items[0].get('id')
            print(f"Found folder '{FOLDER_NAME}' with ID: {self.folder_id}")

    def setup_sheets(self):
        # check if spreadsheet exists inside the folder
        query = f"name='{SHEET_NAME}' and '{self.folder_id}' in parents and trashed=false"
        results = self.drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])

        if not items:
            # Create spreadsheet
            self.spreadsheet = self.gc.create(SHEET_NAME, self.folder_id)
            print(f"Created Spreadsheet '{SHEET_NAME}' with ID: {self.spreadsheet.id}")
            
            # Create sheets
            sheet1 = self.spreadsheet.sheet1
            sheet1.update_title("VoiceClones")
            sheet1.append_row(["ID", "Name", "Ref Audio Drive ID", "Ref Text", "Created At"])
            
            self.spreadsheet.add_worksheet(title="History", rows=100, cols=10)
            history_sheet = self.spreadsheet.worksheet("History")
            history_sheet.append_row(["ID", "Text", "Voice Clone Used", "Status", "Output Audio Drive ID", "Created At"])
        else:
            self.spreadsheet = self.gc.open_by_key(items[0].get('id'))
            print(f"Found Spreadsheet '{SHEET_NAME}' with ID: {self.spreadsheet.id}")
            
            # Ensure worksheets exist
            worksheets = [ws.title for ws in self.spreadsheet.worksheets()]
            if "VoiceClones" not in worksheets:
                ws = self.spreadsheet.add_worksheet("VoiceClones", 100, 10)
                ws.append_row(["ID", "Name", "Ref Audio Drive ID", "Ref Text", "Created At"])
            if "History" not in worksheets:
                ws = self.spreadsheet.add_worksheet("History", 100, 10)
                ws.append_row(["ID", "Text", "Voice Clone Used", "Status", "Output Audio Drive ID", "Created At"])

    # === Drive Operations ===
    def upload_audio(self, file_path, name):
        """Upload an audio file to the Omni Voice folder and return file ID."""
        if not self.drive_service or not self.folder_id:
            return None
        file_metadata = {
            'name': name,
            'parents': [self.folder_id]
        }
        media = MediaFileUpload(file_path, mimetype='audio/wav', resumable=True)
        file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    
    def download_audio(self, file_id, output_path):
        """Download an audio file by file ID to output_path."""
        if not self.drive_service:
            return False
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return True

    # === Voice Clones ===
    def add_voice_clone(self, name, audio_path, ref_text):
        if not self.spreadsheet:
            return None
        audio_id = self.upload_audio(audio_path, f"{name}_ref.wav")
        clone_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ws = self.spreadsheet.worksheet("VoiceClones")
        ws.append_row([clone_id, name, audio_id, ref_text, created_at])
        return clone_id

    def get_voice_clones(self):
        if not self.spreadsheet:
            return pd.DataFrame(columns=["ID", "Name", "Ref Audio Drive ID", "Ref Text", "Created At"])
        ws = self.spreadsheet.worksheet("VoiceClones")
        data = ws.get_all_records()
        return pd.DataFrame(data)
    
    def delete_voice_clone(self, clone_id):
        if not self.spreadsheet:
            return False
        ws = self.spreadsheet.worksheet("VoiceClones")
        records = ws.get_all_records()
        for idx, row in enumerate(records):
            if str(row["ID"]) == str(clone_id):
                # delete from drive as well
                audio_id = row.get("Ref Audio Drive ID")
                if audio_id:
                    try:
                        self.drive_service.files().delete(fileId=audio_id).execute()
                    except Exception:
                        pass
                ws.delete_rows(idx + 2)
                return True
        return False

    # === History ===
    def add_history(self, text, clone_name, status, audio_path=None):
        if not self.spreadsheet:
            return None
        audio_id = ""
        if audio_path and status == "Success":
            audio_id = self.upload_audio(audio_path, f"generated_{uuid.uuid4().hex[:8]}.wav")
        
        history_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ws = self.spreadsheet.worksheet("History")
        ws.append_row([history_id, text, clone_name, status, audio_id, created_at])
        return history_id

    def get_history(self):
        if not self.spreadsheet:
            return pd.DataFrame(columns=["ID", "Text", "Voice Clone Used", "Status", "Output Audio Drive ID", "Created At"])
        ws = self.spreadsheet.worksheet("History")
        data = ws.get_all_records()
        return pd.DataFrame(data)

    def update_history_status(self, history_id, status, audio_path=None):
        if not self.spreadsheet:
            return False
        ws = self.spreadsheet.worksheet("History")
        records = ws.get_all_records()
        for idx, row in enumerate(records):
            if str(row["ID"]) == str(history_id):
                row_idx = idx + 2
                ws.update_cell(row_idx, 4, status)
                if audio_path and status == "Success":
                    audio_id = self.upload_audio(audio_path, f"generated_{uuid.uuid4().hex[:8]}.wav")
                    ws.update_cell(row_idx, 5, audio_id)
                return True
        return False

    def delete_history(self, history_id):
        if not self.spreadsheet:
            return False
        ws = self.spreadsheet.worksheet("History")
        records = ws.get_all_records()
        for idx, row in enumerate(records):
            if str(row["ID"]) == str(history_id):
                audio_id = row.get("Output Audio Drive ID")
                if audio_id:
                    try:
                        self.drive_service.files().delete(fileId=audio_id).execute()
                    except Exception:
                        pass
                ws.delete_rows(idx + 2)
                return True
        return False
