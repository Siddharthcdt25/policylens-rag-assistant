"""Helper utility to generate a synthetic insurance policy PDF for testing.

Creates a 3-page sample policy:
- Page 1: Native text with policy details and SECTION 1 - COVERAGE & DEDUCTIBLES.
- Page 2: Native text with SECTION 2 - EXCLUSIONS and CLAUSE 3.1 WATER DAMAGE.
- Page 3: A scanned-style rider / endorsement page with minimal native text to test OCR fallback.
"""

from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def generate_sample_policy(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    # --- Page 1: Native text policy header and Coverage ---
    page1 = doc.new_page(width=595, height=842)  # A4
    text_p1 = """APEX MUTUAL INSURANCE COMPANY
COMPREHENSIVE PROPERTY POLICY - FORM HO-5
Policy Number: APX-99281-2026
Named Insured: Jordan Ellis
Effective Date: January 1, 2026 to January 1, 2027

SECTION 1 - PROPERTY COVERAGES

Coverage A - Dwelling
We cover the dwelling on the Described Location shown in the Declarations, including structures attached to the dwelling. The limit of liability for Coverage A is $450,000 with a standard deductible of $1,000 per occurrence.

Coverage B - Other Structures
We cover other structures on the Described Location, set apart from the dwelling by clear space. The limit of liability is 10% of Coverage A ($45,000).

Coverage C - Personal Property
We cover personal property owned or used by an Insured while it is anywhere in the world. The standard limit is 50% of Coverage A ($225,000). Special limits apply to jewelry, watches, and furs up to an aggregate limit of $2,500 for loss by theft.

CLAUSE 1.4 - LOSS SETTLEMENT
Covered property losses are settled at Replacement Cost without deduction for depreciation, subject to the condition that the property is repaired or replaced within 180 days of the loss event.
"""
    page1.insert_text((50, 60), text_p1, fontsize=10, fontname="helv")

    # --- Page 2: Exclusions and Conditions ---
    page2 = doc.new_page(width=595, height=842)
    text_p2 = """APEX MUTUAL INSURANCE COMPANY - POLICY FORM HO-5 (CONT.)
Policy Number: APX-99281-2026

SECTION 2 - EXCLUSIONS

We do not insure for loss caused directly or indirectly by any of the following:

2.1 Earth Movement
Earthquake, landslide, mudflow, earth sinking, rising, or shifting, unless direct loss by fire or explosion ensues, and then we will pay only for the ensuing loss.

CLAUSE 2.2 - WATER DAMAGE AND FLOOD
We do not cover loss caused by:
(1) Flood, surface water, waves, tidal water, overflow of any body of water, or spray from any of these, whether or not driven by wind.
(2) Water or water-borne material which backs up through sewers or drains or which overflows or is discharged from a sump, sump pump, or related equipment.
(3) Water below the surface of the ground, including water which exerts pressure on or seeps through a building, sidewalk, driveway, or foundation.

CLAUSE 2.3 - WEAR AND TEAR
Wear and tear, marring, deterioration, latent defect, inherent vice, smog, rust, corrosion, or mechanical breakdown are expressly excluded.

SECTION 3 - DUTIES AFTER LOSS
In case of a loss to covered property, you must give prompt notice to us or our agent, protect the property from further damage, and submit a signed, sworn proof of loss within 60 days after our request.
"""
    page2.insert_text((50, 60), text_p2, fontsize=10, fontname="helv")

    # --- Page 3: Scanned Rider / Endorsement (Simulating a scanned image with no native text) ---
    page3 = doc.new_page(width=595, height=842)
    # Native text is under 50 characters to trigger OCR fallback
    page3.insert_text((50, 40), "ENDORSEMENT R-1", fontsize=8, fontname="helv")

    # Create an image containing text to simulate a scanned document
    img = Image.new("RGB", (1200, 1600), color="white")
    draw = ImageDraw.Draw(img)

    rider_text = (
        "APEX MUTUAL INSURANCE - SPECIAL ENDORSEMENT R-1\n\n"
        "SCHEDULED EQUIPMENT FLOATER\n\n"
        "Clause E-1: High-Value Solar Equipment Coverage\n"
        "It is agreed that photovoltaic solar panels mounted on the roof\n"
        "of the main dwelling are covered up to a maximum limit of $35,000,\n"
        "subject to a separate windstorm deductible of $1,500.\n\n"
        "Clause E-2: Backup of Sewer and Drain Endorsement\n"
        "Notwithstanding Exclusion 2.2, Coverage is provided up to $15,000\n"
        "for direct physical loss caused by water backing up through\n"
        "sewers or drains, with a $500 deductible.\n"
    )
    # Draw text onto image
    draw.text((100, 150), rider_text, fill="black")

    # Save temp image and insert into Page 3
    temp_img_path = output_path.parent / "temp_rider.png"
    img.save(temp_img_path)
    page3.insert_image(fitz.Rect(50, 60, 545, 780), filename=str(temp_img_path))

    doc.save(str(output_path))
    doc.close()

    if temp_img_path.exists():
        temp_img_path.unlink()

    return output_path


if __name__ == "__main__":
    target = Path("data/sample_policy.pdf")
    generate_sample_policy(target)
    print(f"Sample policy generated at: {target.resolve()}")
