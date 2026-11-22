
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


from xml.sax.saxutils import escape


def create_pdf_report(label, confidence, predictions, info):
    buffer = BytesIO()

    # Sanitize strings against ReportLab XML-like markup injection
    safe_label = escape(str(label))
    safe_title = escape(str(info.get("title", "")))
    safe_description = escape(str(info.get("description", "")))
    safe_symptoms = escape(str(info.get("symptoms", "")))

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("PlantCare AI", styles["Title"]))
    story.append(
        Paragraph("Plant Disease Recognition Report", styles["Heading2"])
    )
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 18))

    story.append(Paragraph("Analysis Result", styles["Heading2"]))
    story.append(Paragraph(f"Predicted category: {safe_label}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Model confidence: {confidence * 100:.2f}%",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Top Predictions", styles["Heading2"]))

    table_data = [["Rank", "Category", "Model Score"]]

    for index, prediction in enumerate(predictions, start=1):
        table_data.append(
            [
                str(index),
                escape(str(prediction.get("label", ""))),
                f"{float(prediction.get('score', 0)) * 100:.2f}%",
            ]
        )

    table = Table(table_data, colWidths=[2 * cm, 9 * cm, 4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c1c1c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#fcfbf8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#eceae4")),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Plant Care Information", styles["Heading2"]))
    story.append(Paragraph(safe_title, styles["Heading3"]))
    story.append(Paragraph(safe_description, styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Possible symptoms", styles["Heading3"]))
    story.append(Paragraph(safe_symptoms, styles["Normal"]))
    story.append(Spacer(1, 8))

    treatment_list = info.get("treatment", [])
    prevention_list = info.get("prevention", [])
    care_list = info.get("care", [])

    if treatment_list:
        story.append(Paragraph("How to Treat the Plant (Curative Treatment Protocol)", styles["Heading3"]))
        for item in treatment_list:
            story.append(Paragraph(f"• {escape(str(item))}", styles["Normal"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 6))

    if prevention_list:
        story.append(Paragraph("How to Prevent Disease (Long-Term Prevention Strategy)", styles["Heading3"]))
        for item in prevention_list:
            story.append(Paragraph(f"• {escape(str(item))}", styles["Normal"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 6))

    if not treatment_list and not prevention_list and care_list:
        story.append(Paragraph("General Care & Remediation", styles["Heading3"]))
        for suggestion in care_list:
            story.append(Paragraph(f"• {escape(str(suggestion))}", styles["Normal"]))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Disclaimer: This report contains automated model predictions "
            "and general educational information. It is not a confirmed "
            "diagnosis. Consult an agricultural specialist when needed.",
            styles["Italic"],
        )
    )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()
