# 🔧 Text Overflow Fix - Text Going Outside Borders

## ❌ Problem

Text digits are overflowing outside the border box:

```
┌─────────────────────────┐
│                         │
  12345  678  90  ← Text spills out!
│                         │
└─────────────────────────┘
```

**Cause:** Font size (70px) is too large for the plate width (520px)

---

## ✅ Solution Applied

### **1. Reduced Font Size**
Changed in **Cell 3**:
```python
FONT_SIZE = 55  # Reduced from 70
```

### **2. Added Safety Margins**
```python
BORDER_MARGIN = 5   # Border line position
TEXT_MARGIN = 15    # Text must stay inside this
```

### **3. Added Text Clamping**
In **Cell 6**, text position is now clamped:
```python
# Ensure text NEVER touches borders
x = max(TEXT_MARGIN, min(x, PLATE_WIDTH - text_width - TEXT_MARGIN))
y = max(TEXT_MARGIN, min(y, PLATE_HEIGHT - text_height - TEXT_MARGIN))
```

**Result:** Text is GUARANTEED to fit within borders!

---

## 🔍 How to Verify (Cell 9)

After running Cell 9, check the images:

### **✅ GOOD (Text fits):**
```
┌─────────────────────────┐ ← RED border
│  :::::::::::::::::::::  │ ← BLUE safe zone
│  : 123456  789  01 :   │ ← Text inside BLUE
│  :::::::::::::::::::::  │
└─────────────────────────┘
```

### **❌ BAD (Text overflows):**
```
┌─────────────────────────┐
│123456  789  01│ ← Text touches/crosses RED
└─────────────────────────┘
```

---

## 🎯 Recommended Font Sizes

| Font Size | Text Width | Status | Use Case |
|-----------|------------|--------|----------|
| 70px | ~480px | ⚠️ TOO LARGE | Overflows |
| 60px | ~430px | ⚠️ RISKY | May overflow |
| 55px | ~400px | ✅ GOOD | Current (fits) |
| 50px | ~360px | ✅ SAFE | Guaranteed fit |
| 45px | ~320px | ✅ SAFE | Lots of clearance |

**Plate available width:** 490px (520px - 2×15px margins)

---

## ⚙️ How to Adjust Font Size

### **Method 1: Before Generation (Recommended)**

In **Cell 3**, change:
```python
FONT_SIZE = 50  # Or 45 for guaranteed fit
```

Then re-run cells 3-9.

### **Method 2: Quick Test (Cell 11)**

Uncomment in Cell 11:
```python
FONT_SIZE = 50
font = load_font(FONT_SIZE)
```

Then re-run Cell 9 to verify.

---

## 🐛 Troubleshooting

### **Issue: Text still overflows after reducing font size**

**Possible causes:**

1. **Default font being used** (very wide)
   - Check Cell 5 output
   - Should say: "✅ Font loaded: DejaVuSansMono..."
   - If says: "⚠️ Using PIL default font" → Font not found

**Solution:** Install fonts on Kaggle:
```python
# Add new cell at top:
!apt-get update && apt-get install -y fonts-dejavu-core
```

Then restart notebook.

2. **Font size not updated**
   - Make sure to re-run Cell 5 after changing FONT_SIZE
   - Run: Cell 3 → Cell 5 → Cell 9

3. **Spacing issues**
   - Try reducing spacing between digit groups:
   ```python
   # In Cell 4, change format_plate_text:
   return f'{number[:6]} {number[6:9]} {number[9:11]}'  # Single space
   ```

---

## 📊 Diagnostic Output Explained

### **Cell 9 Output:**

```
📏 Text Measurements:
   Text size: 395×45px          ← Text dimensions
   Plate size: 520×110px        ← Plate dimensions
   Available width: 490px       ← Space inside margins
   Available height: 80px
   ✅ Text fits! Clearance: 95px on sides  ← Good!
```

**What you want to see:**
- ✅ Text width < Available width
- ✅ Clearance > 50px (comfortable)
- ✅ No red warning message

**Bad signs:**
- ❌ "Text is Xpx too wide!"
- ❌ Clearance < 20px (too tight)

---

## 💡 Pro Tips

### **1. Test Before Full Generation**
Always run Cell 9 diagnostics before generating 50,000 images!

### **2. Visual Check**
In Cell 9 images:
- Text should be **clearly inside** BLUE dotted lines
- Should **never touch** RED border lines

### **3. Font Matters**
Different fonts have different widths:
- **DejaVu Sans Mono** (narrow) → Font size 55-60 works
- **Arial/Helvetica** (wider) → Font size 50-55 works
- **PIL Default** (variable) → Font size 45-50 works

### **4. Plate Aspect Ratio**
Current: 520×110 (4.7:1 ratio - Algerian standard)

If you change plate size:
```python
PLATE_WIDTH = 600  # Wider
FONT_SIZE = 65     # Can increase font
```

Formula: `FONT_SIZE ≈ PLATE_WIDTH / 10`

---

## ✅ Final Checklist

Before generating full dataset:

- [ ] Run Cell 9 diagnostics
- [ ] Check "Text Measurements" - should show "✅ Text fits!"
- [ ] Visually verify in images - text inside BLUE zone
- [ ] No text touching or crossing RED border lines
- [ ] Clearance > 50px on each side
- [ ] Font loaded successfully (not using default)

**If all checked → Safe to generate!** 🚀

---

## 📸 Example: Good vs Bad

### **✅ GOOD:**
```
┌──────────────────────────────────┐
│                                  │
│      123456  789  01             │  ← Centered, inside borders
│                                  │
└──────────────────────────────────┘
```

### **❌ BAD (Too Large):**
```
┌──────────────────────────────────┐
│                                  │
 123456  789  01  ← Spills outside!
│                                  │
└──────────────────────────────────┘
```

### **❌ BAD (Touching Border):**
```
┌──────────────────────────────────┐
│123456  789  01│ ← No margin!
└──────────────────────────────────┘
```

---

## 🎯 Quick Fix Summary

1. **Set font size to 50-55px** (Cell 3)
2. **Run Cell 9** to verify
3. **Check visual guides** (text inside BLUE zone)
4. **If still overflows** → Try font size 45px
5. **If using default font** → Install proper fonts

**Current settings (Cell 3):**
- FONT_SIZE = 55px ✅
- TEXT_MARGIN = 15px ✅
- Border clamping enabled ✅

**This should work! If not, reduce to 50px.** 👍
