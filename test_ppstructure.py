"""
Test PP-Structure Table Recognition API
========================================
Jalankan: python test_ppstructure.py
"""
import sys
import os

print("=" * 60)
print("  Testing PP-Structure Table Recognition API")
print("=" * 60)

# ─── 1. Check imports ─────────────────────────────────────────
print("\n1. Checking imports...")

paddlex_available = False
ppstructure_available = False

try:
    from paddlex import create_pipeline
    paddlex_available = True
    print("   ✅ paddlex.create_pipeline available")
except ImportError as e:
    print(f"   ❌ paddlex not available: {e}")

try:
    from paddleocr import PPStructure
    ppstructure_available = True
    print("   ✅ PPStructure from paddleocr available")
except ImportError as e:
    print(f"   ❌ PPStructure not available: {e}")

if not paddlex_available and not ppstructure_available:
    print("\n❌ No table recognition API available!")
    print("   Try: pip install paddlex")
    sys.exit(1)

# ─── 2. Test on image ────────────────────────────────────────
img = sys.argv[1] if len(sys.argv) > 1 else "termudah_ocrtest.jpg"
print(f"\n2. Testing on: {img}")

if not os.path.exists(img):
    print(f"   ❌ File not found: {img}")
    sys.exit(1)

# ─── 2a. Try paddlex pipeline ────────────────────────────────
if paddlex_available:
    print("\n── paddlex.create_pipeline('table_recognition') ──")
    try:
        pipeline = create_pipeline(pipeline="table_recognition")
        output = pipeline.predict(input=img)
        
        for i, result in enumerate(output):
            print(f"\n   Result [{i}]:")
            print(f"   Type: {type(result).__name__}")
            
            # Check for dict-like result
            if hasattr(result, '__dict__'):
                attrs = [k for k in result.__dict__.keys() if not k.startswith('_')]
                print(f"   Attributes: {attrs}")
            
            if isinstance(result, dict):
                print(f"   Keys: {list(result.keys())}")
                for k, v in result.items():
                    v_str = str(v)[:200]
                    print(f"      {k}: {v_str}")
            
            # Try common attribute names
            for attr in ['html', 'table_html', 'cell_bbox', 'cell_texts', 
                         'rec_texts', 'dt_polys', 'table_res']:
                if hasattr(result, attr):
                    val = getattr(result, attr)
                    val_str = str(val)[:300]
                    print(f"   .{attr}: {val_str}")
            
            # Try to print full result
            result_str = str(result)[:500]
            print(f"   str(result): {result_str}")
            
            # If result has items() method
            if hasattr(result, 'items'):
                for k, v in result.items():
                    v_str = str(v)[:200]
                    print(f"   [{k}]: {v_str}")

            # Check for save methods
            save_methods = [m for m in dir(result) if 'save' in m.lower() or 'html' in m.lower()]
            if save_methods:
                print(f"   Save/HTML methods: {save_methods}")
                    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ─── 2b. Try PPStructure ─────────────────────────────────────
if ppstructure_available:
    print("\n── PPStructure() ──")
    try:
        engine = PPStructure(show_log=False, lang="en")
        result = engine(img)
        
        if not result:
            print("   ❌ No results")
        else:
            for i, item in enumerate(result):
                print(f"\n   Item [{i}]:")
                print(f"   Type: {type(item).__name__}")
                
                if isinstance(item, dict):
                    print(f"   Keys: {list(item.keys())}")
                    for k, v in item.items():
                        v_str = str(v)[:200]
                        print(f"      {k}: {v_str}")
                else:
                    print(f"   Content: {str(item)[:300]}")
                    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("  Test selesai!")
print("=" * 60)
