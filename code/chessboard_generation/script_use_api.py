import subprocess
import os
import csv
import glob
import time
from PIL import Image

# --- Configuration ---
# CRITICAL: Use 'blender.exe', NOT 'blender-launcher.exe'.
# CRITICAL: Change for your own directory for those files, for some reason relative directory don't work
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE = r"C:\Users\yoavl\Documents\github\Introduction_to_deep_learning_project\code\chessboard_generation\chess-set.blend"
SCRIPT_FILE = r"C:\Users\yoavl\Documents\github\Introduction_to_deep_learning_project\code\chessboard_generation\chess_position_api_v2.py"
RESOLUTION = 2400  # Resolution defined here to use in crop calculations


def generate_dataset():
    # 1. Get input directory
    input_dir_raw = input("Enter the directory path containing the CSV file: ").strip().strip('"')
    input_dir = os.path.abspath(input_dir_raw)

    # 2. Find the CSV file
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No .csv file found in {input_dir}")
        return

    csv_path = csv_files[0]
    print(f"Found CSV: {csv_path}")

    # 3. Setup Output Directory (Automatic)
    output_dir = os.path.join(input_dir, "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    # 4. Select View to Keep and Setup Crop Logic
    print("\nThe Blender script generates 3 views. Which one do you want to keep?")
    print("1. overhead (Keeps '1_overhead.png') - Cropped Center")
    print("2. east     (Keeps '2_east.png')     - Cropped Right Side")
    print("3. west     (Keeps '3_west.png')     - Cropped Left Side")

    view_map = {
        "1": ("1_overhead.png", "overhead"),
        "overhead": ("1_overhead.png", "overhead"),
        "2": ("2_east.png", "east"),
        "east": ("2_east.png", "east"),
        "3": ("3_west.png", "west"),
        "west": ("3_west.png", "west")
    }

    choice = input("Enter choice: ").strip().lower()
    target_source_file, view_type = view_map.get(choice, ("1_overhead.png", "overhead"))
    print(f"Selected view: {view_type} (Source: {target_source_file})")

    # Define all generated files to cleanup later
    generated_files = ["1_overhead.png", "2_east.png", "3_west.png"]

    # 5. Iterate through CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        print(1)
        for i, row in enumerate(reader):
            try:
                frame_id = int(row['from_frame'])
                fen = row['fen']
            except KeyError:
                print("Error: CSV must have 'from_frame' and 'fen' columns.")
                return

            final_filename = f"frame_{frame_id:06d}.jpg"
            final_path = os.path.join(output_dir, final_filename)

            if os.path.exists(final_path):
                print(f"Skipping frame {frame_id}, file exists.")
                continue

            print(f"Generating frame {frame_id}...")

            # 6. Construct Blender Command
            cmd = [
                BLENDER_EXE,
                BLEND_FILE,
                "--background",
                "--python", SCRIPT_FILE,
                "--",
                "--fen", fen,
                "--resolution", str(RESOLUTION),
                "--output_path", output_dir,
                "--view", "white",
                "--view_angle", view_type
            ]

            if i == 0:
                print(f"\n[DEBUG] Running command:\n{' '.join(cmd)}\n")

            # 7. Run Blender
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Blender failed for frame {frame_id}.")
                print(f"--- ERROR LOG (STDERR) ---\n{e.stderr[-500:]}")
                continue

            # 8. Wait for file generation (Retry Loop)
            source_path = os.path.join(output_dir, target_source_file)
            max_retries = 3000
            found = False

            for attempt in range(max_retries):
                if os.path.exists(source_path):
                    found = True
                    break
                time.sleep(1)

            # 9. Crop, Save, and Cleanup
            if found:
                time.sleep(0.5)  # Buffer for file release

                try:
                    # Open original image
                    with Image.open(source_path) as img:
                        width, height = img.size  # Should be RESOLUTION (800)

                        # --- CROP LOGIC START ---
                        # Box tuple: (left, top, right, bottom)
                        if view_type == "overhead":
                            # x, y both from 0.4 to 0.6
                            left = width * 0.4
                            top = height * 0.4
                            right = width * 0.6
                            bottom = height * 0.6
                            crop_box = (left, top, right, bottom)

                        elif view_type == "west":
                            # y same (full height?), x from res/16 to res/16 + res/10
                            # Assuming "y same" implies keeping original Y or specific ratio?
                            # Standard interpretation: keep full height or same as overhead?
                            # Instruction said "if east then the y the same".
                            # Ambiguous if "same as original" or "same as overhead".
                            # Context implies "same as original" if not specified,
                            # BUT "y the same" likely refers to the overhead crop logic?
                            # Re-reading: "if overhead... crop y... if east y the same" -> likely same crop on Y.

                            # Y Crop (Same as overhead):
                            top = height * 0.4
                            bottom = height * 0.6

                            # X Crop:
                            left = width / 16
                            right = (width / 16) + (2*(width / 10))
                            crop_box = (left, top, right, bottom)

                        elif view_type == "east":
                            # Y Crop (Same as overhead):
                            top = height * 0.4
                            bottom = height * 0.6

                            # X Crop:
                            # from res - res/16 - res/10 to res - res/16
                            right = width - (width / 16)
                            left = width - (width / 16) - (2*(width / 10))

                            crop_box = (left, top, right, bottom)
                        # --- CROP LOGIC END ---

                        cropped_img = img.crop(crop_box)

                        # Save directly to final path (converting to RGB for JPG)
                        if os.path.exists(final_path):
                            os.remove(final_path)
                        cropped_img.convert("RGB").save(final_path, "JPEG")

                except Exception as e:
                    print(f"Error processing image for frame {frame_id}: {e}")

                # Cleanup all generated source files
                for fname in generated_files:
                    path_to_remove = os.path.join(output_dir, fname)
                    if os.path.exists(path_to_remove):
                        try:
                            #os.remove(path_to_remove)
                            a = 1
                        except OSError:
                            pass
            else:
                print(
                    f"Warning: Expected file {target_source_file} not found in {output_dir} after {max_retries} seconds.")

    print("\nGeneration process complete.")


if __name__ == "__main__":
    generate_dataset()