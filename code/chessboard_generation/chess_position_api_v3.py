"""
Chess FEN Parser - Auto-detect Starting Positions

Strategy:
1. Analyze current piece positions to determine which piece is on which square
2. Parse target FEN
3. Move pieces from their detected starting square to target square

Usage:
    blender chess-set.blend --background --python chess_position_api_v2.py -- --fen "r4rk1/1p1bqppp/n1p1pn2/p2pN3/2PP4/P1N3P1/1P1QPPBP/R4RK1" --view black
"""
from email.policy import default
import random
import bpy
import math
from mathutils import Vector
import sys
import argparse
# Rotate the offset vector
from mathutils import Matrix
# ==========================
# CONFIG
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 2
DESIRED_ANGLE_DEGREES = 25
LENS = 26
RES = 1024
SAMPLES = 128
OUT_DIR = "//renders"

def get_board_info():
    """Get board dimensions"""
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    
    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8
    
    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)
    
    scale_factor = board_size / REAL_BOARD_SIZE
    
    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'scale_factor': scale_factor,
    }

def position_to_square(pos, board_info):
    """Convert 3D position to chess square (e.g., 'e2')"""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']
    
    # File (a-h) from X coordinate - scene is flipped
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    # Rank (1-8) from Y coordinate
    # Higher Y = lower rank (reversed)
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1
    
    return f"{file_letter}{rank_number}"

def detect_starting_positions(board_info):
    """
    Detect which piece is on which square currently
    Returns: {piece_name: {'square': 'e2', 'piece_type': 'P'}}
    """
    print("\n" + "="*70)
    print("DETECTING STARTING POSITIONS")
    print("="*70)
    
    pieces = {}
    
    # Get all chess piece objects
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        
        name = obj.name
        
        # Determine piece type from name
        piece_type = None
        
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']:
            piece_type = 'P'  # White pawn
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']:
            piece_type = 'p'  # Black pawn
        elif 'rook' in name.lower():
            piece_type = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower():
            piece_type = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower():
            piece_type = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower():
            piece_type = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower():
            piece_type = 'K' if 'white' in name.lower() else 'k'

        if piece_type:
            square = position_to_square(obj.location, board_info)
            pieces[name] = {
                'square': square,
                'piece_type': piece_type,
                'start_pos': obj.location.copy()
            }
            print(f"  {name:20s} → {square:4s} ({piece_type})")
    
    print(f"\n✓ Detected {len(pieces)} pieces")
    return pieces

def parse_fen(fen):
    """Parse FEN into dict {square: piece_char}"""
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')
    
    position = {}
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx
        
        for char in rank:
            if char.isdigit():
                file_idx += int(char)
            else:
                file_letter = chr(ord('a') + file_idx)
                square = f"{file_letter}{board_rank}"
                position[square] = char
                file_idx += 1
    
    return position

def apply_fen(fen, starting_pieces, board_info):
    """Apply FEN by moving pieces from detected starting positions"""
    print("\n" + "="*70)
    print("APPLYING FEN")
    print("="*70)
    print(f"FEN: {fen}\n")
    
    target_position = parse_fen(fen)
    square_size = board_info['square_size']
    
    # Build reverse mapping: start_square+piece_type -> piece_name
    available_pieces = {}
    for piece_name, info in starting_pieces.items():
        key = (info['square'], info['piece_type'])
        if key not in available_pieces:
            available_pieces[key] = []
        available_pieces[key].append(piece_name)
    
    pieces_used = set()
    
    # For each target square
    for target_square, piece_type in target_position.items():
        # Find a piece of this type
        candidates = []
        for piece_name, info in starting_pieces.items():
            if info['piece_type'] == piece_type and piece_name not in pieces_used:
                # Calculate distance
                from_square = info['square']
                from_file = ord(from_square[0]) - ord('a')
                from_rank = int(from_square[1]) - 1
                to_file = ord(target_square[0]) - ord('a')
                to_rank = int(target_square[1]) - 1
                
                distance = abs(to_file - from_file) + abs(to_rank - from_rank)
                candidates.append((distance, piece_name, from_square))
        
        piece_name = None
        from_square = None

        # LOGIC: Choose existing piece OR duplicate if none available
        if not candidates:
            # Try to duplicate a piece of this type
            duplicate_candidates = [name for name, info in starting_pieces.items() if info['piece_type'] == piece_type]
            
            if duplicate_candidates:
                original_name = duplicate_candidates[0]
                original_obj = bpy.data.objects.get(original_name)
                
                if original_obj:
                    # COPY THE OBJECT (Data API)
                    new_obj = original_obj.copy()
                    new_obj.data = original_obj.data 
                    new_obj.animation_data_clear()
                    
                    # LINK TO COLLECTION (Crucial for rendering)
                    if original_obj.users_collection:
                        original_obj.users_collection[0].objects.link(new_obj)
                    else:
                        bpy.context.scene.collection.objects.link(new_obj)
                    
                    # Update internal tracking
                    new_name = new_obj.name
                    starting_pieces[new_name] = {
                        'square': position_to_square(new_obj.location, board_info),
                        'piece_type': piece_type,
                        'start_pos': new_obj.location.copy()
                    }
                    
                    piece_name = new_name
                    from_square = starting_pieces[piece_name]['square']
                    print(f"  + Duplicated {original_name} -> {new_name}")
            else:
                print(f"  Warning: No piece of type '{piece_type}' available for {target_square}")
                continue
        else:
            # Use closest existing piece
            candidates.sort()
            _, piece_name, from_square = candidates[0]
        
        # --- MOVE THE PIECE ---
        if piece_name:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                # Calculate movement
                from_file = ord(from_square[0]) - ord('a')
                from_rank = int(from_square[1]) - 1
                to_file = ord(target_square[0]) - ord('a')
                to_rank = int(target_square[1]) - 1
                
                file_diff = to_file - from_file
                rank_diff = to_rank - from_rank
                
                # Reset to start pos first (to avoid cumulative error)
                start_pos = starting_pieces[piece_name]['start_pos']
                obj.location = start_pos.copy()
                
                # Apply Move
                obj.location.x -= file_diff * square_size
                obj.location.y -= rank_diff * square_size
                
                # SHOW piece
                obj.hide_render = False
                obj.hide_viewport = False
                
                pieces_used.add(piece_name)
                
                if from_square != target_square:
                    print(f"  Moved {piece_name:20s} {from_square} -> {target_square}")
                else:
                    print(f"  Kept {piece_name:20s} at {target_square}")

    # Hide unused pieces
    for piece_name in starting_pieces.keys():
        if piece_name not in pieces_used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True
                # Do NOT print hidden logs to keep output clean
    
    # Force Board Visibility (Fix for missing board)
    board = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    if board: board.hide_render = False
    if frame: frame.hide_render = False

    print(f"\n✓ Position set ({len(pieces_used)} pieces visible)")

def render_all_views(board_info, view='black'):
    """Render views from white or black perspective"""
    print("\n" + "="*70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("="*70)
    
    center = board_info['center']
    scale_factor = board_info['scale_factor']
    
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    horizontal_offset = camera_height * math.tan(angle_radians)
    
    # Clean cameras
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Setup lighting
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light_height = center.z + camera_height * 2
        bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
        bpy.context.active_object.data.energy = 3.0
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = 'PNG'
    scene.cycles.use_denoising = True
    
    try:#starts
        # Access Cycles preferences
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        
        # Set compute device type to OPTIX (Best for RTX 4050)
        cycles_prefs.compute_device_type = 'OPTIX'
        
        # Refresh device list to ensure Blender sees the cards
        cycles_prefs.get_devices()
        
        # Iterate through devices: Enable NVIDIA, Disable Intel/CPU
        for device in cycles_prefs.devices:
            if "NVIDIA" in device.name:
                device.use = True
                print(f"  [ENABLED]  {device.name}")
            else:
                device.use = False
        
        # Apply GPU settings to the scene
        scene.cycles.device = 'GPU'
    except Exception as e:
        print(f"  [ERROR] GPU Setup Failed: {e}")
        pass#finish
    
    # Camera positions
    camera_z = center.z + camera_height
    
    # Flip camera positions for white view (180 degree rotation)
    if view == 'white':
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x + horizontal_offset, center.y, camera_z), "2_east", False),
            ((center.x - horizontal_offset, center.y, camera_z), "3_west", False),
        ]
        z_rotation_offset = math.radians(180)
    else:  # black view (default)
        views = [
            ((center.x, center.y, camera_z), "1_overhead", True),
            ((center.x - horizontal_offset, center.y, camera_z), "2_west", False),
            ((center.x + horizontal_offset, center.y, camera_z), "3_east", False),
        ]
        z_rotation_offset = 0
    
    for location, name, point_at_center in views:
        if(VIEW_ANGLE in name):
            print(f"\nRendering: {name}")

            bpy.ops.object.camera_add(location=location)
            cam = bpy.context.active_object

            if point_at_center:
                direction = center - cam.location
                cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            else:
                cam.rotation_euler = (0, 0, 0)

            # Apply rotation for white/black view
            cam.rotation_euler.z += z_rotation_offset

            cam.data.lens = LENS

            bpy.context.scene.camera = cam
            bpy.context.scene.render.filepath = f"{OUT_DIR}/{name}.png"
            bpy.ops.render.render(write_still=True)

            print(f"  ✓ Saved: {name}.png")

            bpy.data.objects.remove(cam, do_unlink=True)
    
    print("\n✓ Rendering complete")

def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    parser.add_argument('--resolution', type=int, default=800)
    parser.add_argument('--output_path', type=str, default="./renders")
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'],
                        help='Render from white or black perspective')
    parser.add_argument('--view_angle', type=str, default='overhead', choices=['overhead', 'east', 'west'],)
    
    args = parser.parse_args(argv)
    
    global RES, SAMPLES, OUT_DIR, VIEW_ANGLE
    RES = args.resolution
    SAMPLES = args.samples
    OUT_DIR = args.output_path
    VIEW_ANGLE = args.view_angle

    
    # Get board info
    board_info = get_board_info()
    # Fix inverted board - rotate checkerboard 90 degrees around board center
    plane = bpy.data.objects.get("Black & white")
    if plane:
        # Get board center first (before rotating)
        frame = bpy.data.objects.get("Outer frame")
        frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
        frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
        frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
        center = (frame_min + frame_max) / 2
        
        # Store original position
        original_pos = plane.location.copy()
        
        # Move to center, rotate, move back
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)
        
        rot_matrix = Matrix.Rotation(math.radians(90), 3, 'Z')
        rotated_offset = rot_matrix @ offset
        
        plane.location = center + rotated_offset
    # Detect starting positions
    starting_pieces = detect_starting_positions(board_info)
    
    # Apply FEN
    apply_fen(args.fen, starting_pieces, board_info)
    
    # Render
    render_all_views(board_info, view=args.view)

if __name__ == "__main__":
    main()
