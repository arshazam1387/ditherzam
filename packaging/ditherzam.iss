#define MyAppName "ditherzam"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "ditherzam contributors"
#define MyAppURL "https://github.com/arshazam1387/ditherzam"
#define MyAppExeName "ditherzam.exe"

[Setup]
AppId={{6A92065F-DCE9-4D5F-B454-F9314D0980A8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile=..\LICENSE
InfoBeforeFile=..\THIRD_PARTY_NOTICES.md
OutputDir=..\release
OutputBaseFilename=ditherzam-0.1.0-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}
VersionInfoVersion=0.1.0.0
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion=0.1.0
VersionInfoDescription={#MyAppName} installer
VersionInfoCompany={#MyAppPublisher}
VersionInfoCopyright=GPL-3.0-only

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\ditherzam\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ditherzam"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\License and notices"; Filename: "{app}\THIRD_PARTY_NOTICES.md"
Name: "{autodesktop}\ditherzam"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch ditherzam"; Flags: nowait postinstall skipifsilent unchecked
