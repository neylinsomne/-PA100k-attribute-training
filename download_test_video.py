"""
Download test video from YouTube for attribute testing
Uses yt-dlp to download a short video with people for testing attribute detection
"""
import subprocess
import sys
from pathlib import Path

# Video URL (YouTube short with people)
VIDEO_URL = "https://www.youtube.com/shorts/hxeudw4U8Cw"
OUTPUT_DIR = Path(__file__).parent / "test_videos"
OUTPUT_FILE = OUTPUT_DIR / "attributes_sim.mp4"

def check_ytdlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ytdlp():
    """Install yt-dlp using pip"""
    print("Installing yt-dlp...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp", "--user"],
            check=True
        )
        print("yt-dlp installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing yt-dlp: {e}")
        return False

def download_video():
    """Download video from YouTube"""
    print(f"\nDownloading test video from YouTube...")
    print(f"URL: {VIDEO_URL}")
    print(f"Output: {OUTPUT_FILE}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Download video
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "best[ext=mp4]/best",  # Best MP4 format
                "-o", str(OUTPUT_FILE),
                VIDEO_URL
            ],
            check=True
        )

        print(f"\n✅ Video downloaded successfully!")
        print(f"   Location: {OUTPUT_FILE}")
        print(f"   Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading video: {e}")
        return False

def main():
    print("=" * 70)
    print("Download Test Video for Attribute Detection")
    print("=" * 70)

    # Check if video already exists
    if OUTPUT_FILE.exists():
        print(f"\n✅ Video already exists: {OUTPUT_FILE}")
        print(f"   Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

        overwrite = input("\nDo you want to download again? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Skipping download.")
            return

    # Check/install yt-dlp
    if not check_ytdlp():
        print("\nyt-dlp not found.")
        install = input("Do you want to install it? (y/n): ").strip().lower()

        if install == 'y':
            if not install_ytdlp():
                print("\n❌ Failed to install yt-dlp")
                print("Please install manually: pip install yt-dlp")
                sys.exit(1)
        else:
            print("\n❌ yt-dlp is required to download videos")
            print("Install with: pip install yt-dlp")
            sys.exit(1)

    # Download video
    if download_video():
        print("\n" + "=" * 70)
        print("Video ready for testing!")
        print("=" * 70)
        print("\nTo test attributes, run:")
        print(f"  python test_attributes_cpu.py")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
