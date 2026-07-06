"""Generate placeholder sample images for evaluation (no API required)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def _font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def receipt() -> None:
    img = Image.new("RGB", (400, 520), "white")
    d = ImageDraw.Draw(img)
    f, fs = _font(16), _font(14)
    lines = [
        "COFFEE CORNER",
        "123 Main St",
        "----------------",
        "Latte        $4.50",
        "Muffin       $3.25",
        "----------------",
        "Subtotal     $7.75",
        "Tax          $0.62",
        "TOTAL        $8.37",
        "Date: 2026-03-15",
    ]
    y = 20
    for line in lines:
        d.text((30, y), line, fill="black", font=fs if line.startswith(("Latte", "Muffin", "Sub")) else f)
        y += 28
    img.save(OUT / "receipt_sample.png")


def whiteboard() -> None:
    img = Image.new("RGB", (500, 400), "#f5f5dc")
    d = ImageDraw.Draw(img)
    f = _font(18)
    title = "Sprint Planning"
    d.text((20, 20), title, fill="navy", font=f)
    bullets = [
        "- Ship auth by Friday",
        "- Alex: API review",
        "- Book demo with client",
        "- ACTION: send timeline",
    ]
    y = 60
    for b in bullets:
        d.text((30, y), b, fill="black", font=_font(16))
        y += 36
    img.save(OUT / "whiteboard_sample.png")


def chart() -> None:
    img = Image.new("RGB", (480, 320), "white")
    d = ImageDraw.Draw(img)
    d.text((140, 10), "Q1 Revenue", fill="black", font=_font(18))
    bars = [("Jan", 80), ("Feb", 120), ("Mar", 95)]
    x = 60
    for label, h in bars:
        d.rectangle([x, 280 - h, x + 60, 280], fill="steelblue")
        d.text((x + 15, 290), label, fill="black", font=_font(14))
        x += 100
    d.line([(40, 280), (440, 280)], fill="black", width=2)
    img.save(OUT / "chart_sample.png")


def equipment() -> None:
    img = Image.new("RGB", (420, 300), "#e8e8e8")
    d = ImageDraw.Draw(img)
    d.rectangle([80, 60, 340, 220], fill="#555", outline="black", width=2)
    d.text((100, 100), "PUMP UNIT A-12", fill="yellow", font=_font(16))
    d.text((100, 140), "Pressure: 42 PSI", fill="white", font=_font(14))
    d.text((100, 170), "WARNING: leak detected", fill="red", font=_font(14))
    img.save(OUT / "equipment_sample.png")


def screenshot() -> None:
    img = Image.new("RGB", (450, 280), "#1e1e1e")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 450, 40], fill="#333")
    d.text((15, 10), "Project Dashboard", fill="white", font=_font(16))
    d.text((20, 60), "Status: On track", fill="#4ade80", font=_font(14))
    d.text((20, 90), "Tasks open: 7", fill="white", font=_font(14))
    d.text((20, 120), "Next milestone: Apr 1", fill="white", font=_font(14))
    img.save(OUT / "screenshot_sample.png")


def main() -> None:
    receipt()
    whiteboard()
    chart()
    equipment()
    screenshot()
    print(f"Created 5 sample images in {OUT}")


if __name__ == "__main__":
    main()
