"""Disease information catalog, clinical pathology, treatment protocols, and prevention strategies.

Contains authoritative reference data for all 38 PlantVillage / PlantCare AI classes
spanning 14 agricultural crops, plus robust matching logic and thumbnail base64 encoding.
"""

from __future__ import annotations

import base64
import functools
import re
from pathlib import Path
from typing import Any

DISEASE_INFO: dict[str, dict[str, Any]] = {
    # ---------------- 1. APPLE (4 Classes) ----------------
    "Apple___Apple_scab": {
        "title": "Apple Scab (Venturia inaequalis)",
        "crop": "Apple",
        "pathogen": "Fungus (Venturia inaequalis)",
        "severity": "Moderate",
        "description": "Apple scab is a pervasive fungal disease caused by Venturia inaequalis. It attacks both foliar tissues and emerging fruit, inducing olive-green to velvet-brown velvety lesions that lead to premature defoliation, stunted fruit development, and corky surface cracking.",
        "symptoms": "Olive-drab to sooty dark green velvety spots on the upper leaf surface, progressing into puckered, distorted leaf margins and premature summer drop.",
        "treatment": [
            "Prune infected twigs and branches during dry weather to eliminate overwintering fungal pseudothecia.",
            "Apply targeted bio-fungicides such as Copper Octanoate, Liquid Copper, or Myclobutanil at green tip through petal fall.",
            "Use Bacillus amyloliquefaciens (Serenade ASO) bio-spray at 5-7 day intervals during sustained leaf wetness periods.",
            "Collect and incinerate or deeply compost all fallen leaf litter under the canopy immediately."
        ],
        "prevention": [
            "Plant scab-resistant cultivars such as Liberty, Enterprise, Freedom, GoldRush, or Prima.",
            "Maintain wide canopy spacing with annual dormant thinning to maximize sunlight penetration and wind drying.",
            "Avoid overhead irrigation; utilize ground-level drip emitters to keep foliage dry during spore-release hours.",
            "Apply a 5% urea foliar spray in late autumn just before leaf drop to accelerate leaf decomposition and exhaust ascospores."
        ],
    },
    "Apple___Black_rot": {
        "title": "Apple Black Rot (Botryosphaeria obtusa)",
        "crop": "Apple",
        "pathogen": "Fungus (Botryosphaeria obtusa)",
        "severity": "High",
        "description": "Black rot (also called frog-eye leaf spot) is a destructive fungal pathogen causing leaf spotting, perennial limb cankers, and firm, concentric black fruit rot. Spores overwinter in dead wood, mummified apples, and bark crevices.",
        "symptoms": "Frog-eye foliar lesions with pale tan or grayish centers bounded by sharp purple-brown margins, progressing to branch dieback and mummified dark fruit.",
        "treatment": [
            "Excise and prune all dead, blighted twigs, branch cankers, and mummified fruit at least 15 cm below infected margins.",
            "Disinfect pruning shears between each cut using 70% isopropyl alcohol or a 10% sodium hypochlorite solution.",
            "Apply protective fungicides containing captan, thiophanate-methyl, or sulfur starting at tight cluster through mid-summer.",
            "Remove all mummies and pruned wood from the orchard vicinity and destroy immediately."
        ],
        "prevention": [
            "Prune annual growth annually to minimize inner canopy humidity and prevent limb rubbing wounds.",
            "Protect trunks and limbs against winter injury, sunburn, and mechanical weed-whacker lacerations.",
            "Maintain optimal soil fertility and tree vigor; avoid over-fertilizing with high nitrogen which encourages vulnerable succulent growth.",
            "Monitor nearby cedar or juniper trees and wild rosaceous brush for potential fungal spore bridges."
        ],
    },
    "Apple___Cedar_apple_rust": {
        "title": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "crop": "Apple",
        "pathogen": "Fungus (Gymnosporangium juniperi-virginianae)",
        "severity": "Moderate",
        "description": "A heteroecious rust fungus requiring two alternating hosts: Eastern red cedar (Juniperus virginiana) and apple/crabapple trees. In spring, gelatinous orange telial horns erupt on cedar galls, broadcasting aeciospores onto emerging apple foliage.",
        "symptoms": "Bright yellow-orange or fiery reddish-orange circular spots on the upper leaf surfaces, developing tubular aecia (tube-like spore cups) underneath that exude powdery spores.",
        "treatment": [
            "Apply protective fungicides (Myclobutanil, Mancozeb, or Immunox) beginning at pink bud stage until 1-2 weeks after petal fall.",
            "Foliar sprays of potassium bicarbonate or wettable sulfur can be used as organic knockdowns when infection conditions are observed.",
            "Pick off and bag isolated rusted leaves early in the season if the infection is confined to lower canopy branches.",
            "Apply curative systemic triazole sprays within 48 hours of suspected spring spore-flight rain events."
        ],
        "prevention": [
            "Remove or cull nearby Eastern red cedar, ornamental junipers, or galls within a 1-mile perimeter where feasible.",
            "Plant resistant apple varieties such as Redfree, Freedom, Liberty, McIntosh, or William's Pride.",
            "Prune cedar galls from ornamental landscape junipers in late winter before gelatinous spring telial horns emerge.",
            "Space orchard rows in line with prevailing prevailing winds to accelerate canopy drying."
        ],
    },
    "Apple___healthy": {
        "title": "Healthy Apple Foliage (Malus domestica)",
        "crop": "Apple",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Foliage exhibits vigorous deep-green chlorophyll density, intact serrated leaf margins, uniform cellular turgor, and zero fungal or bacterial lesions.",
        "symptoms": "No visible pathology. Symmetrical venation, robust cuticle layer, and vibrant dark green photosynthetic coloration.",
        "treatment": [
            "No curative intervention required; plant tissue is healthy and thriving.",
            "Maintain baseline soil moisture levels through dry periods with drip irrigation.",
            "Inspect weekly for early aphid, mite, or foliar rust incursions."
        ],
        "prevention": [
            "Maintain balanced N-P-K soil nutrition with supplemental micronutrients (Zinc, Boron, Calcium).",
            "Perform regular winter sanitation and dormant oil sprays to smother overwintering scale and mite eggs.",
            "Maintain 8-10 cm of organic woodchip mulch around root zones, keeping mulch 15 cm away from the trunk collar."
        ],
    },

    # ---------------- 2. BLUEBERRY (1 Class) ----------------
    "Blueberry___healthy": {
        "title": "Healthy Highbush Blueberry (Vaccinium corymbosum)",
        "crop": "Blueberry",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Highbush blueberry leaves demonstrate rich green coloration, smooth leathery leaf texture, active photosynthetic capacity, and optimal acidic soil balance.",
        "symptoms": "Intact leaves without chlorotic interveinal striping, anthracnose spots, or leaf scorch.",
        "treatment": [
            "No chemical or biological treatment needed.",
            "Maintain acidic soil moisture (pH 4.5 - 5.2) using elemental sulfur or acidifying organic amendments.",
            "Protect against summer heat stress with adequate drip line irrigation."
        ],
        "prevention": [
            "Mulch heavily with pine bark or acidic conifer sawdust (8-12 cm depth) to preserve root moisture and suppress weeds.",
            "Avoid alkaline tap water or nitrate-heavy fertilizers; supply nitrogen exclusively in ammonium form.",
            "Prune older unproductive canes older than 6 years during winter dormancy."
        ],
    },

    # ---------------- 3. CHERRY (2 Classes) ----------------
    "Cherry_(including_sour)___Powdery_mildew": {
        "title": "Cherry Powdery Mildew (Podosphaera clandestina)",
        "crop": "Cherry",
        "pathogen": "Fungus (Podosphaera clandestina)",
        "severity": "Moderate",
        "description": "An obligate fungal parasite affecting sweet and sour cherries. White fungal mycelial mats colonize the underside of tender expanding leaves, leading to upward rolling, chlorotic mottling, and unmarketable fruit blemishes.",
        "symptoms": "Circular patches of powdery white or gray mycelium on the underside of young leaves, causing leaves to cup upward and become crisp and brittle.",
        "treatment": [
            "Apply potassium bicarbonate (Kaligreen, MilStop) or horticultural mineral oil (1-2%) at first detection of powdery patches.",
            "Rotate synthetic fungicides such as Quinoxyfen, Fluopyram, or Tebuconazole to prevent resistance development.",
            "Prune out heavily infested shoots and sucker sprouts that harbor dense mycelium.",
            "Maintain protective spray cover during late spring warm, humid, non-rainy conditions."
        ],
        "prevention": [
            "Improve orchard canopy ventilation through selective summer pruning of water sprouts.",
            "Avoid high nitrogen applications in late spring, which produces vulnerable succulent foliar tissue.",
            "Utilize drip or microsprinkler irrigation; avoid wetting the canopy with high-angle overhead irrigation.",
            "Monitor lower inner branches early in the season when temperatures range between 18°C and 28°C."
        ],
    },
    "Cherry_(including_sour)___healthy": {
        "title": "Healthy Cherry Foliage (Prunus avium / cerasus)",
        "crop": "Cherry",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Foliage displays deep lustrous green pigment, smooth serrate margins, vigorous petiole attachment, and freedom from fungal shot-hole or powdery mildew.",
        "symptoms": "No visible disease. Clean foliar lamina, uniform chlorophyll reflectance, and robust photosynthetic health.",
        "treatment": [
            "No curative intervention necessary.",
            "Maintain regular soil moisture during fruit swell and post-harvest bud differentiation.",
            "Monitor shoot tips for black cherry aphid colonization."
        ],
        "prevention": [
            "Apply dormant copper spray at autumn leaf drop and spring bud swell to safeguard against bacterial canker.",
            "Paint lower tree trunks with 50/50 white latex paint and water to prevent winter sunscald bark cracking.",
            "Prune sweet cherries exclusively during dry summer periods to inhibit cytospora and bacterial canker entry."
        ],
    },

    # ---------------- 4. CORN / MAIZE (4 Classes) ----------------
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "title": "Corn Gray Leaf Spot (Cercospora zeae-maydis)",
        "crop": "Corn",
        "pathogen": "Fungus (Cercospora zeae-maydis)",
        "severity": "High",
        "description": "One of the most economically devastating foliar diseases in commercial maize production. Fungal hyphae germinate from previous crop debris under extended cloud cover and high relative humidity, invading leaves through stomata and disrupting photosynthetic grain fill.",
        "symptoms": "Distinctive rectangular, vein-delimited tan-to-gray lesions running parallel with the leaf veins, expanding up to 5-10 cm in length and blighting whole leaves.",
        "treatment": [
            "Apply strobilurin (azoxystrobin, pyraclostrobin) or triazole (propiconazole) fungicides at VT-R1 (tasseling to silking) stage if lesions appear on the third leaf below the ear.",
            "Ensure thorough spray coverage of the upper ear leaves, which contribute over 75% of photosynthetic grain fill.",
            "Rotate fungicide FRAC groups (FRAC 11 + FRAC 3) to prevent fungal resistance.",
            "Harvest heavily affected plots early to avert stalk lodging and grain loss."
        ],
        "prevention": [
            "Plant corn hybrids with high Cercospora genetic tolerance ratings.",
            "Implement a minimum 2-year crop rotation with non-host crops like soybeans, wheat, or alfalfa.",
            "Utilize conservation tillage or deep fall tillage to accelerate decomposition of infected corn stover residue.",
            "Avoid excessive plant densities that impede canopy air movement and prolong morning dew duration."
        ],
    },
    "Corn_(maize)___Common_rust_": {
        "title": "Corn Common Rust (Puccinia sorghi)",
        "crop": "Corn",
        "pathogen": "Fungus (Puccinia sorghi)",
        "severity": "Moderate",
        "description": "Foliar fungal disease caused by Puccinia sorghi. Urediniospores are transported by high-altitude southern wind currents from overwintering regions in the subtropics, forming pustules on both upper and lower leaf surfaces during cool, humid conditions.",
        "symptoms": "Golden-brown to cinnamon-brown powdery circular to elongated pustules erupting through the leaf epidermis on both leaf surfaces, turning brownish-black (teliospores) in autumn.",
        "treatment": [
            "Apply foliar fungicides (Pyraclostrobin, Azoxystrobin + Difenoconazole) if rust pustules cover over 5% of leaf area prior to silking.",
            "Repeat spray applications within 14 days if cool, damp, overcast weather persists and disease pressure is severe.",
            "Target sweet corn and seed production corn fields with higher sensitivity compared to commercial dent field corn."
        ],
        "prevention": [
            "Plant resistant corn hybrids containing the Rp1-D single-gene resistance or general polygenic rust resistance.",
            "Plant early in the growing season to ensure grain fill occurs prior to peak late-summer spore arrival.",
            "Maintain balanced soil potassium and nitrogen; excessive nitrogen promotes lush foliar tissue favored by rust fungi."
        ],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "title": "Corn Northern Corn Leaf Blight (Exserohilum turcicum)",
        "crop": "Corn",
        "pathogen": "Fungus (Exserohilum turcicum / Setosphaeria turcica)",
        "severity": "High",
        "description": "A serious foliar blight causing significant yield reduction when established prior to silking. The fungus overwinters in corn residue and produces conidia that germinate under moderate temperatures (18°C–27°C) and prolonged wetness periods.",
        "symptoms": "Large, elongated, cigar-shaped grayish-green to tan necrotic lesions (3 to 15 cm long) with smooth margins, developing olive-brown fungal spores across lesions in humid weather.",
        "treatment": [
            "Apply dual-action fungicides (QoI + DMI modes of action, such as Headline AMP or Trivapro) when lesions reach threshold levels on ear leaves at VT-R1.",
            "Apply treatments using high-clearance ground sprayers with drop nozzles for optimal ear-zone penetration.",
            "Ensure prompt silage harvesting if widespread premature canopy dieback occurs."
        ],
        "prevention": [
            "Select corn varieties with multi-gene (Ht1, Ht2, Ht3, or HtN) resistance traits.",
            "Rotate fields out of corn for at least 1-2 seasons to break the inoculum cycle in the soil and crop debris.",
            "Shred and incorporate corn residue post-harvest with chisel plowing or discing to accelerate microbial degradation."
        ],
    },
    "Corn_(maize)___healthy": {
        "title": "Healthy Field / Sweet Corn (Zea mays)",
        "crop": "Corn",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Corn leaves exhibit uniform chlorophyll saturation, clear leaf venation, intact leaf collars, and optimal nitrogen assimilation without nutrient deficiency or foliar blights.",
        "symptoms": "No disease symptoms. Robust midrib, clean margins, and bright uniform dark green color.",
        "treatment": [
            "No curative intervention required.",
            "Maintain side-dress nitrogen management matching crop growth stages (V6-V8).",
            "Scout weekly for rootworms, corn borers, and foliar spotting."
        ],
        "prevention": [
            "Employ certified high-germination seed with standard fungicide and insecticide seed coatings.",
            "Calibrate seeding depth (3.8 - 5.0 cm) and plant into soils above 10°C for rapid seedling vigor.",
            "Implement rotational cropping with legumes to build organic matter and soil health."
        ],
    },

    # ---------------- 5. GRAPE (4 Classes) ----------------
    "Grape___Black_rot": {
        "title": "Grape Black Rot (Guignardia bidwellii)",
        "crop": "Grape",
        "pathogen": "Fungus (Guignardia bidwellii)",
        "severity": "Critical",
        "description": "One of the most destructive diseases of cultivated grapes in warm, humid regions. Capable of causing 100% crop loss through foliar blighting and complete desiccation of fruit clusters into hard, wrinkled black mummies.",
        "symptoms": "Small circular reddish-brown leaf spots with dark margins, studded with tiny black pycnidia (spore fruiting bodies); berries turn soft, brown, and shrink into hard black mummies.",
        "treatment": [
            "Apply protective fungicides (Mancozeb, Captan, Myclobutanil) starting at 2.5 cm shoot growth, repeated at 10-14 day intervals through 4 weeks post-bloom.",
            "Remove and incinerate all mummified grape clusters remaining on trellises and the vineyard floor during dormant pruning.",
            "Spray with copper hydroxide or sulfur during organic production cycles before infection rain periods.",
            "Prune out shoots displaying black elongated canker lesions."
        ],
        "prevention": [
            "Practice open canopy training systems (Vertical Shoot Positioning - VSP) to maximize air movement and solar exposure.",
            "Orient vineyard rows parallel to prevailing winds to hasten foliage drying after rain or heavy dew.",
            "Till or cover vineyard floor with clean mulch to bury overwintering spore reservoirs.",
            "Plant less susceptible cultivars or hybrids where black rot pressure is endemic."
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "title": "Grape Esca / Black Measles (Phaeomoniella & Fomitiporia complex)",
        "crop": "Grape",
        "pathogen": "Fungal Complex (Phaeomoniella chlamydospora & Fomitiporia mediterranea)",
        "severity": "Critical",
        "description": "A complex vascular grapevine wood disease affecting mature vineyards. The fungi colonize xylem vessels, producing phytotoxic metabolites that induce characteristic tiger-stripe foliar interveinal necrosis and apoplexy (sudden vine collapse).",
        "symptoms": "'Tiger-stripe' foliar chlorosis and necrosis where yellow and brown zones form between green leaf veins, accompanied by small dark purple spots on fruit ('measles').",
        "treatment": [
            "No single chemical cure exists for systemic vascular infection; surgical remedial trunk renewal or trunk surgery is required.",
            "Cut trunks down to healthy, symptomless white wood below internal rot column and train a renewal sucker.",
            "Mark symptomatic vines in summer for delayed dormant pruning to avoid spreading spores to healthy vines.",
            "Seal all large pruning wounds immediately with wound paint or biological biocontrol pastes containing Trichoderma."
        ],
        "prevention": [
            "Adopt double pruning or late pruning (late February/March) when wound healing is rapid and spore dispersal is lower.",
            "Avoid making large flush pruning cuts close to the main trunk; leave a protective wood collar.",
            "Sanitize pruning shears with 70% alcohol or quaternary ammonium solutions between vines.",
            "Purchase certified disease-tested planting material from reputable certified vine nurseries."
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "title": "Grape Leaf Blight (Pseudocercospora vitis / Isariopsis)",
        "crop": "Grape",
        "pathogen": "Fungus (Pseudocercospora vitis)",
        "severity": "Moderate",
        "description": "Foliar blight disease causing irregular dark brown angular lesions on grape foliage late in the season. Severe infection leads to extensive premature defoliation, reducing cluster sugar accumulation and impairing vine cold hardiness.",
        "symptoms": "Irregular, angular, dark reddish-brown to black spots (2-8 mm) on leaf blades, which may coalesce into large blighted zones with olive-brown sporulation on the underside.",
        "treatment": [
            "Apply copper-based fungicides (Bordeaux mixture, Copper Hydroxide) or Mancozeb at the first sign of late-season spotting.",
            "Maintain protective fungicide cover during post-harvest warm, wet conditions to protect canopy foliage.",
            "Gather and destroy fallen blighted leaves to reduce overwintering spore survival."
        ],
        "prevention": [
            "Maintain an open trellis canopy with shoot thinning, leaf pulling, and hedging to reduce interior humidity.",
            "Avoid excessive late-season nitrogen fertilization that delays wood hardening and encourages late vegetative flushes.",
            "Utilize sub-canopy drip irrigation rather than overhead sprinklers."
        ],
    },
    "Grape___healthy": {
        "title": "Healthy Grapevine Foliage (Vitis vinifera)",
        "crop": "Grape",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Grape leaves exhibit broad, vigorous lobed blades with rich green pigmentation, intact petioles, and robust photosynthetic gas exchange without necrotic spotting or powdery mildew.",
        "symptoms": "No visible pathology. Clean leaf surfaces, intact venation, and strong shoot vigor.",
        "treatment": [
            "No chemical intervention required.",
            "Maintain shoot positioning and cane tie-downs along trellis support wires.",
            "Monitor weekly for early downy mildew oil spots or leafhopper activity."
        ],
        "prevention": [
            "Implement routine canopy leaf thinning in the fruiting zone to optimize sun penetration and berry ventilation.",
            "Conduct seasonal petiole tissue analysis to guide balanced potassium, magnesium, and boron fertilization.",
            "Sow perennial cover crops between vineyard rows to promote soil microbiology and suppress dust."
        ],
    },

    # ---------------- 6. CITRUS / ORANGE (1 Class) ----------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "title": "Citrus Huanglongbing / Greening (Candidatus Liberibacter asiaticus)",
        "crop": "Citrus",
        "pathogen": "Bacterium (Candidatus Liberibacter asiaticus)",
        "severity": "Critical",
        "description": "Also known as HLB or yellow dragon disease, this is the most devastating citrus disease globally. Transmitted by the Asian citrus psyllid (Diaphorina citri), the phloem-restricted bacterium causes vascular clogging, root starvation, lopsided bitter fruit, and tree death.",
        "symptoms": "Blotchy, asymmetrical yellow foliar mottle crossing leaf veins (unlike symmetrical nutritional deficiencies), thickened raised veins, twig dieback, and small misshapen bitter fruit with inverted ripening.",
        "treatment": [
            "No curative cure exists once systemic infection occurs; remove and eradicate confirmed positive trees to stop orchard transmission.",
            "Aggressively suppress the vector: apply systemic and foliar insecticides (Imidacloprid, Thiamethoxam, Bifenthrin) targeting psyllid flushes.",
            "Introduce biological parasitoid wasps (Tamarixia radiata) in biocontrol management zones.",
            "Support declining infected trees with foliar nutritional cocktail sprays (zinc, manganese, iron, potassium phosphite) to temporarily extend productive lifespan."
        ],
        "prevention": [
            "Plant certified pathogen-free citrus stock grown strictly in insect-proof screened nurseries.",
            "Monitor yellow sticky traps weekly to detect Asian citrus psyllid arrivals.",
            "Eradicate non-commercial host plants like Orange Jasmine (Murraya paniculata) from neighboring boundaries.",
            "Comply with international and regional plant quarantine regulations prohibiting interstate citrus movement."
        ],
    },

    # ---------------- 7. PEACH (2 Classes) ----------------
    "Peach___Bacterial_spot": {
        "title": "Peach Bacterial Spot (Xanthomonas arboricola pv. pruni)",
        "crop": "Peach",
        "pathogen": "Bacterium (Xanthomonas arboricola pv. pruni)",
        "severity": "High",
        "description": "A major bacterial pathogen of stone fruits. Bacteria enter through stomata and wind-blown sand or hail wounds during warm, rainy springs, causing shot-hole foliar perforation, severe defoliation, and cracked necrotic fruit lesions.",
        "symptoms": "Small, water-soaked, angular purple-brown spots on leaves that dry out and drop out, creating a 'shot-hole' appearance; tips of leaves turn yellow and drop prematurely.",
        "treatment": [
            "Apply protective copper bactericide sprays at dormant leaf fall and delayed dormant bud swell.",
            "Use Oxytetracycline (Mycoshield) or copper octanoate sprays at low rates from petal fall through cover sprays at 7-10 day intervals.",
            "Prune out overwintered spring twig cankers before bloom to lower inoculum density.",
            "Avoid high copper rates in warm weather to prevent chemical phytotoxicity and leaf burn."
        ],
        "prevention": [
            "Plant resistant peach cultivars such as Candor, Clayton, Dixired, Harvester, Norman, or Winblo.",
            "Establish windbreaks in sandy soil regions to minimize wind-driven sand abrasion and wounds.",
            "Maintain moderate tree vigor; avoid excessive late nitrogen fertilization.",
            "Use trickle or drip irrigation; keep sprinklers positioned beneath tree canopies."
        ],
    },
    "Peach___healthy": {
        "title": "Healthy Peach Foliage (Prunus persica)",
        "crop": "Peach",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Peach foliage displays vibrant lanceolate leaves with lustrous emerald green coloration, intact glandular serrations, and active shoot elongation without curl, shot-hole, or yellowing.",
        "symptoms": "No visible pathology. Smooth leaf surfaces, uniform light reflection, and supple healthy petioles.",
        "treatment": [
            "No chemical intervention needed.",
            "Maintain soil moisture during stone hardening and fruit swell.",
            "Scout for early signs of oriental fruit moth or peach tree borer."
        ],
        "prevention": [
            "Apply a single dormant copper or chlorothalonil spray before bud swell to prevent Peach Leaf Curl.",
            "Maintain an open-center (vase) canopy architecture to maximize sunlight and air circulation.",
            "Keep orchard floor clear of fallen mummified fruit and fallen leaves."
        ],
    },

    # ---------------- 8. PEPPER, BELL (2 Classes) ----------------
    "Pepper,_bell___Bacterial_spot": {
        "title": "Bell Pepper Bacterial Spot (Xanthomonas campestris pv. vesicatoria)",
        "crop": "Pepper",
        "pathogen": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "severity": "High",
        "description": "A destructive seed-borne bacterial pathogen affecting bell and hot peppers. Spread rapidly by wind-driven rain, overhead sprinklers, and field handling of wet foliage, resulting in extensive leaf drop, sunscald on unprotected fruit, and unmarketable scabby fruit lesions.",
        "symptoms": "Small, water-soaked circular to angular lesions (1-3 mm) turning brownish-black with a greasy appearance on undersides of leaves, causing rapid yellowing and severe defoliation.",
        "treatment": [
            "Apply copper hydroxide tank-mixed with mancozeb to overcome copper-tolerant bacterial strains at 7-day intervals.",
            "Spray bacteriophage biological products (such as AgriPhage) during active early disease outbreaks.",
            "Apply Actigard (acibenzolar-S-methyl) as a systemic acquired resistance (SAR) inducer.",
            "Remove and safely destroy heavily infected plants from small greenhouse or garden settings."
        ],
        "prevention": [
            "Plant certified disease-free, hot-water-treated seed and resistant pepper hybrids (races 1-10 resistance).",
            "Strictly avoid working in or cultivating pepper rows when foliage is wet from rain or dew.",
            "Use drip irrigation exclusively; avoid all forms of overhead sprinkler irrigation.",
            "Enforce a 2 to 3 year crop rotation away from solanaceous crops (tomatoes, peppers, potatoes, eggplants)."
        ],
    },
    "Pepper,_bell___healthy": {
        "title": "Healthy Bell Pepper Foliage (Capsicum annuum)",
        "crop": "Pepper",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Foliage exhibits lush, smooth, ovate-to-lanceolate leaves with intense green chlorophyll density, upright turgor pressure, and vigorous branching without chlorosis or necrotic spotting.",
        "symptoms": "No disease indicators. Clean uniform green lamina, intact veins, and active terminal bud growth.",
        "treatment": [
            "No curative intervention required.",
            "Maintain consistent soil moisture to prevent blossom end rot.",
            "Support heavy fruiting plants with stakes or trellis lines."
        ],
        "prevention": [
            "Apply balanced fertilizer with adequate calcium and magnesium.",
            "Apply organic straw or plastic mulch to suppress weeds and stabilize soil temperature.",
            "Maintain 45-60 cm spacing between plants for good airflow."
        ],
    },

    # ---------------- 9. POTATO (3 Classes) ----------------
    "Potato___Early_blight": {
        "title": "Potato Early Blight (Alternaria solani)",
        "crop": "Potato",
        "pathogen": "Fungus (Alternaria solani)",
        "severity": "Moderate",
        "description": "A prevalent fungal pathogen targeting older foliage under alternating dry and wet conditions. The fungus causes premature leaf senescence and significant reduction in tuber size and yield, overwintering on infested plant residue.",
        "symptoms": "Dark brown to black circular-to-irregular lesions displaying distinctive concentric target-like rings, surrounded by a narrow chlorotic yellow halo on lower mature leaves.",
        "treatment": [
            "Apply protective fungicides containing chlorothalonil, mancozeb, or copper at initial lower canopy spotting.",
            "Alternate with translaminar systemic fungicides (such as Boscalid, Difenoconazole, or Azoxystrobin) to prevent resistance.",
            "Ensure complete coverage of lower mature foliage where fungal spores initiate infection.",
            "Maintain adequate nitrogen and potassium fertility; stressed plants are far more susceptible to early blight."
        ],
        "prevention": [
            "Select potato varieties with higher field tolerance (such as Ranger Russet, Russet Burbank).",
            "Employ a 3-4 year crop rotation avoiding solanaceous crops like tomato, eggplant, and pepper.",
            "Irrigate early in the morning so foliar surfaces dry completely before nightfall.",
            "Destroy crop haulms and vines prior to harvest; allow tuber skins to set firmly before digging."
        ],
    },
    "Potato___Late_blight": {
        "title": "Potato Late Blight (Phytophthora infestans)",
        "crop": "Potato",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Critical",
        "description": "The notorious historical agent of the Irish Potato Famine. Phytophthora infestans is an extremely aggressive oomycete water-mold capable of destroying entire fields within days under cool, foggy, wet weather (10°C–21°C and relative humidity >90%).",
        "symptoms": "Large, irregular, water-soaked pale to dark olive-brown lesions that rapidly expand and turn black and purplish; delicate white fungal-like downy growth develops on the lesion undersides in high humidity.",
        "treatment": [
            "Immediately apply curative systemic oomycide fungicides such as Mandipropamid (Revus), Cymoxanil (Curzate), or Fluopicolide.",
            "Apply high-dose protective sprays (Chlorothalonil, Mancozeb) on uninfected surrounding rows.",
            "In organic farming, apply concentrated copper hydroxide or copper sulfate immediately upon local disease forecasting alerts.",
            "Destroy and bury or haul out severely blighted hot spots to prevent aerial spore plumes from contaminating regional fields."
        ],
        "prevention": [
            "Plant only certified disease-free seed tubers; never plant volunteer potatoes or cull pile leftovers.",
            "Eliminate all potato cull piles by burying under 60 cm of soil or spreading and freezing.",
            "Hilling soil generously (10-15 cm) over tuber beds to create a physical filter preventing spores from washing into tubers.",
            "Kill potato vines chemically or mechanically 2-3 weeks before harvest; never dig tubers when foliage is green and wet."
        ],
    },
    "Potato___healthy": {
        "title": "Healthy Potato Foliage (Solanum tuberosum)",
        "crop": "Potato",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Compound potato foliage is robust, deep forest green, and actively photosynthesizing to build tuber carbohydrate reserves, free from fungal spotting or virus mottling.",
        "symptoms": "No visible pathology. Sturdy compound leaves, vibrant green leaflets, and vigorous vegetative canopy.",
        "treatment": [
            "No curative intervention required.",
            "Maintain soil moisture at 70-80% available water capacity during tuber initiation and bulking.",
            "Scout weekly for Colorado potato beetle larvae or aphids."
        ],
        "prevention": [
            "Maintain loose, well-drained soil rich in organic matter with pH 5.5 - 6.5.",
            "Ensure hilling is completed before full canopy closure to protect tubers from greening and pests.",
            "Rotate crops annually with brassicas, cereals, or legumes."
        ],
    },

    # ---------------- 10. RASPBERRY (1 Class) ----------------
    "Raspberry___healthy": {
        "title": "Healthy Raspberry Foliage (Rubus idaeus)",
        "crop": "Raspberry",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Raspberry primocanes and floricanes show vibrant compound green leaves, healthy spines, and clean margins without spur blight, rust, or anthracnose lesions.",
        "symptoms": "No disease symptoms. Intact trifoliate/pinnate leaves, clean white underside pubescence, and strong cane vigor.",
        "treatment": [
            "No chemical intervention needed.",
            "Maintain trellis support wires to keep canes off the ground.",
            "Irrigate through base drip lines during cane elongation and fruit ripening."
        ],
        "prevention": [
            "Prune out old fruited floricanes immediately after summer harvest to increase sunlight and airflow.",
            "Thin primocanes to 4-6 strong canes per linear foot of row.",
            "Apply clean organic woodchip mulch to suppress weeds and retain soil moisture."
        ],
    },

    # ---------------- 11. SOYBEAN (1 Class) ----------------
    "Soybean___healthy": {
        "title": "Healthy Soybean Foliage (Glycine max)",
        "crop": "Soybean",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Soybean trifoliate foliage displays uniform rich green pigment, vigorous nodulation on roots, and clean leaf surfaces free from frogeye leaf spot, rust, or downy mildew.",
        "symptoms": "No disease symptoms. Smooth trifoliate leaflets, intact venation, and balanced photosynthetic growth.",
        "treatment": [
            "No chemical intervention required.",
            "Maintain field scouting for soybean aphids or bean leaf beetles during R1-R3 stages.",
            "Ensure consistent moisture during pod filling (R4-R5)."
        ],
        "prevention": [
            "Inoculate seeds with Bradyrhizobium japonicum before planting in new fields.",
            "Select soybean varieties with multi-race Phytophthora resistance (Rps genes) and cyst nematode tolerance.",
            "Rotate crops annually with corn, grain sorghum, or wheat."
        ],
    },

    # ---------------- 12. SQUASH (1 Class) ----------------
    "Squash___Powdery_mildew": {
        "title": "Squash / Cucurbit Powdery Mildew (Podosphaera xanthii)",
        "crop": "Squash",
        "pathogen": "Fungus (Podosphaera xanthii / Erysiphe cichoracearum)",
        "severity": "Moderate",
        "description": "A very common fungal disease affecting squash, pumpkins, cucumbers, and melons. Fungal colonies establish on both leaf surfaces under high humidity, shaded canopies, and moderate temperatures, leading to premature leaf scorch, reduced yields, and sunburned fruit.",
        "symptoms": "Talcum powder-like white circular fungal patches on the upper and lower leaf surfaces, leaf petioles, and stems; infected leaves eventually turn yellow, brown, and dry out crisp.",
        "treatment": [
            "Apply potassium bicarbonate (MilStop, Armicarb) or horticultural neem oil at the first sign of white spots.",
            "Apply bio-fungicides containing Bacillus amyloliquefaciens or sulfur-based organic sprays every 7-10 days.",
            "Spray systemic fungicides (such as Quinoxyfen, Penthiopyrad, or Triflumizole) rotating FRAC groups.",
            "Prune out older, heavily infected lower leaves to increase ventilation."
        ],
        "prevention": [
            "Plant powdery mildew-resistant (PMR) squash and cucurbit cultivars.",
            "Provide ample spacing (90-120 cm between plants) to reduce canopy crowding and improve air circulation.",
            "Plant in full sun; powdery mildew thrives in shaded or filtered light conditions.",
            "Use ground drip irrigation to keep foliage dry and minimize spore germination."
        ],
    },

    # ---------------- 13. STRAWBERRY (2 Classes) ----------------
    "Strawberry___Leaf_scorch": {
        "title": "Strawberry Leaf Scorch (Diplocarpon earlianum)",
        "crop": "Strawberry",
        "pathogen": "Fungus (Diplocarpon earlianum)",
        "severity": "Moderate",
        "description": "A widespread fungal foliar disease of cultivated strawberries. The pathogen infects leaves, petioles, runners, and calyxes, causing numerous purple-to-brown lesions that coalesce and cause entire leaves to curl upward and look scorched by heat.",
        "symptoms": "Numerous small, dark purple to reddish-brown irregular spots (1-5 mm) without white centers; as spots multiply, leaf tissues between spots turn red, purple, then brown and dry up.",
        "treatment": [
            "Apply protective fungicides (Captan, Thiram, or Pyraclostrobin + Boscalid) beginning at early spring crown growth.",
            "Apply copper-based organic sprays during early spring emergence.",
            "Prune off and discard severely scorched leaves during spring bed cleanup.",
            "Mow or renovate strawberry beds immediately after harvest to remove infected foliage."
        ],
        "prevention": [
            "Plant certified disease-free strawberry crowns from reliable nurseries.",
            "Select resistant cultivars such as Allstar, Earliglow, Honeoye, or Tribute.",
            "Space plants adequately to facilitate rapid foliar drying after morning dew.",
            "Utilize drip tape underneath straw or plastic mulch rather than overhead sprinklers."
        ],
    },
    "Strawberry___healthy": {
        "title": "Healthy Strawberry Foliage (Fragaria × ananassa)",
        "crop": "Strawberry",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Strawberry plants show trifoliate leaves with vibrant, glossy dark green color, distinct serrations, sturdy upright petioles, and active crown runner production without foliar spotting.",
        "symptoms": "No visible pathology. Glossy deep green foliage, robust crowns, and clean flower trusses.",
        "treatment": [
            "No curative intervention required.",
            "Maintain soil moisture at 2.5 cm per week, especially during flowering and fruit sizing.",
            "Protect developing fruit clusters with clean straw mulch."
        ],
        "prevention": [
            "Plant in well-drained, raised beds with sandy loam soil enriched with organic compost (pH 6.0 - 6.8).",
            "Renovate June-bearing beds annually after harvest by thinning crowns and fertilizing.",
            "Inspect weekly for spider mites and crown rot under the leaf canopy."
        ],
    },

    # ---------------- 14. TOMATO (10 Classes) ----------------
    "Tomato___Bacterial_spot": {
        "title": "Tomato Bacterial Spot (Xanthomonas vesicatoria / perforans)",
        "crop": "Tomato",
        "pathogen": "Bacterium (Xanthomonas vesicatoria / perforans / gardneri)",
        "severity": "High",
        "description": "A severe bacterial disease impacting tomatoes globally in warm, humid regions with frequent rainfall. The bacteria enter stomata and wounds, causing widespread leaf defoliation, blossom drop, and raised scabby lesions on green and ripe fruit.",
        "symptoms": "Small, water-soaked, dark brown to black circular lesions (less than 3 mm) on leaves, often surrounded by a faint yellow halo; leaves turn brown, wither, and drop, leaving fruit exposed to sunscald.",
        "treatment": [
            "Apply fixed copper bactericides tank-mixed with Mancozeb at 5-7 day intervals during wet warm weather.",
            "Utilize bacteriophage sprays (AgriPhage) applied during late afternoon or overcast periods for targeted biological control.",
            "Apply Actigard (acibenzolar-S-methyl) to activate systemic acquired plant immunity.",
            "Sterilize all pruning stakes, shears, and trellising twine with 10% bleach or 70% ethanol."
        ],
        "prevention": [
            "Plant certified disease-free seed or hot-water-treated seed (50°C for 25 minutes).",
            "Grow resistant tomato cultivars (e.g. varieties with Rx-3 or Xv-4 resistance genes).",
            "Never cultivate or harvest tomato plants when foliage is wet from rain or dew.",
            "Implement a 3-year crop rotation strictly avoiding all other solanaceous crops."
        ],
    },
    "Tomato___Early_blight": {
        "title": "Tomato Early Blight (Alternaria solani)",
        "crop": "Tomato",
        "pathogen": "Fungus (Alternaria solani / Alternaria linariae)",
        "severity": "Moderate",
        "description": "A ubiquitous fungal pathogen affecting tomatoes. Overwinters on infected crop debris and in the soil, splashing onto bottom foliage during rain events and causing progressive defoliation from the base upward.",
        "symptoms": "Dark brown to black circular spots on older lower leaves featuring distinctive concentric rings ('bullseye' target pattern), surrounded by a bright chlorotic yellow margin.",
        "treatment": [
            "Remove and destroy infected lower leaves up to 45 cm from the soil line to halt upward fungal progression.",
            "Apply bio-fungicides (Bacillus subtilis, copper octanoate) or chemical fungicides (Chlorothalonil, Azoxystrobin, Mancozeb) at 7-10 day intervals.",
            "Prune sucker growth to improve airflow and facilitate thorough spray penetration into the interior canopy.",
            "Ensure adequate potassium and nitrogen nutrition to alleviate plant stress."
        ],
        "prevention": [
            "Apply a 5-8 cm layer of organic straw or black plastic mulch to form a barrier stopping soil-borne spores from splashing onto foliage.",
            "Stake or cage tomato plants to elevate all foliage off the soil surface.",
            "Water exclusively at ground level using drip irrigation or soaker hoses; keep foliage completely dry.",
            "Follow a 3-year crop rotation away from solanaceous plants (potatoes, peppers, eggplants)."
        ],
    },
    "Tomato___Late_blight": {
        "title": "Tomato Late Blight (Phytophthora infestans)",
        "crop": "Tomato",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Critical",
        "description": "An extremely aggressive, fast-moving oomycete pathogen capable of killing healthy tomato plants and turning whole crops into brown necrotic mush within 7 to 10 days during cool, rainy, humid conditions (15°C–22°C).",
        "symptoms": "Large, irregular, water-soaked greenish-black lesions on leaves and stems that expand rapidly and turn greasy dark brown; in humid conditions, a fine white downy fungal-like bloom appears on lesion undersides.",
        "treatment": [
            "Immediately spray systemic anti-oomycete compounds such as Mandipropamid (Revus), Cymoxanil (Curzate), or Dimethomorph.",
            "In organic production, apply liquid copper hydroxide or copper sulfate immediately upon regional blight alerts.",
            "Prune and destroy infected plants immediately; bag infected tissue on-site before moving to prevent airborne spore drift.",
            "Destroy and bury all infected plant debris; do not add to home compost heaps."
        ],
        "prevention": [
            "Plant resistant tomato hybrids carrying Ph-2 and Ph-3 resistance genes (e.g., Mountain Magic, Defiant PhR, Iron Lady).",
            "Monitor regional late blight forecasting networks (e.g. USAblight alerts) to apply preventive sprays ahead of storm fronts.",
            "Ensure broad plant spacing (60-90 cm) in full sun with optimal air drainage.",
            "Destroy all volunteer tomato and potato plants emerging in or near the field early in spring."
        ],
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato Leaf Mold (Passalora fulva / Cladosporium fulvum)",
        "crop": "Tomato",
        "pathogen": "Fungus (Passalora fulva)",
        "severity": "Moderate",
        "description": "A prevalent foliar fungal disease common in high tunnels, greenhouses, and humid sheltered garden plots where relative humidity exceeds 85%. Spores attack foliage, causing premature defoliation and severe yield loss.",
        "symptoms": "Pale green to bright yellow chlorotic patches on the upper surface of older leaves, accompanied by velvety olive-green to grayish-brown spore mats directly underneath on the lower leaf surface.",
        "treatment": [
            "Prune infected lower foliage immediately to reduce spore load and open canopy air pathways.",
            "Apply bio-fungicides (Bacillus amyloliquefaciens, potassium bicarbonate) or copper fungicides at weekly intervals.",
            "In high tunnels, open ridge vents, roll up sidewalls, and run horizontal circulation fans to drive relative humidity below 80%.",
            "Clean and sanitize greenhouse framework and trellising posts with sanitizing disinfectant post-season."
        ],
        "prevention": [
            "Select leaf mold-resistant greenhouse tomato cultivars (hybrids with Cf-1 through Cf-9 resistance genes).",
            "Maintain wide plant spacing and aggressively prune suckers to optimize canopy ventilation.",
            "Avoid late evening overhead watering; keep greenhouse foliage completely dry during nightfall.",
            "Heat and ventilate greenhouses in early morning and late evening to purge humid air."
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Tomato Septoria Leaf Spot (Septoria lycopersici)",
        "crop": "Tomato",
        "pathogen": "Fungus (Septoria lycopersici)",
        "severity": "Moderate",
        "description": "One of the most destructive foliar diseases of field-grown tomatoes. The fungus survives in overwintered solanaceous crop debris and weeds (horse nettle), producing pycnidia that release conidia splashed upward onto lower leaves during rainfall.",
        "symptoms": "Numerous small, circular spots (1.5 - 3 mm) with dark brown margins and sunken grayish-white centers, peppered with tiny black specks (pycnidia) visible under magnification; causes progressive upward leaf yellowing and leaf loss.",
        "treatment": [
            "Prune off infected lower leaves as soon as first spots are detected on bottom foliage.",
            "Apply protective fungicides (Chlorothalonil, Mancozeb, Copper Hydroxide) on a 7-day schedule during rainy warm weather.",
            "Apply organic bio-fungicides containing Bacillus subtilis or copper soap.",
            "Gather and remove all fallen diseased leaves from around the base of the plants."
        ],
        "prevention": [
            "Apply a clean mulch layer (straw, paper, or plastic) around plants immediately after transplanting to block soil splash.",
            "Eradicate solanaceous weeds (such as black nightshade and horse nettle) within and around the garden.",
            "Use drip irrigation or water at the base of the plant in the morning.",
            "Enforce a 2-3 year crop rotation away from tomatoes, potatoes, and eggplants."
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Tomato Two-Spotted Spider Mite (Tetranychus urticae)",
        "crop": "Tomato",
        "pathogen": "Arachnid Pest (Tetranychus urticae)",
        "severity": "Moderate",
        "description": "Microscopic arachnid pests that proliferate rapidly in hot (above 30°C), dry, dusty conditions. Spider mites pierce leaf cells with their stylets and suck out cell contents, causing severe stippling, chlorosis, and fine silken webbing across foliage.",
        "symptoms": "Fine pale yellow or white speckling (stippling) on upper leaf surfaces; dense fine silken webs covering leaf undersides and growing tips; heavily infested leaves turn bronze, dry out, and drop.",
        "treatment": [
            "Spray plants forcefully with a cold water hose blast to dislodge mites and destroy webs.",
            "Apply insecticidal soap or horticultural oil (1-2% solution), ensuring thorough drenching of leaf undersides.",
            "Apply selective miticides (such as Bifenazate, Abamectin, or Spiromesifen) rotating chemistries to avoid rapid mite resistance.",
            "Release predatory mites (Phytoseiulus persimilis or Neoseiulus californicus) into greenhouses or high tunnels."
        ],
        "prevention": [
            "Keep farm roads and garden walkways moist or mulched to suppress dust, which promotes mite explosions.",
            "Avoid broad-spectrum synthetic pyrethroid insecticides that eliminate natural predatory beneficial insects (lacewings, ladybugs, predatory mites).",
            "Maintain optimal plant hydration; drought-stressed tomatoes have higher amino acid concentrations preferred by mites.",
            "Scout leaf undersides with a 10x-14x hand lens weekly during hot summer months."
        ],
    },
    "Tomato___Target_Spot": {
        "title": "Tomato Target Spot (Corynespora cassiicola)",
        "crop": "Tomato",
        "pathogen": "Fungus (Corynespora cassiicola)",
        "severity": "Moderate",
        "description": "A warm-climate fungal disease of field tomatoes. Corynespora cassiicola thrives in humid tropical and subtropical environments, infecting leaves, stems, and fruit, causing severe defoliation and sunken fruit rot.",
        "symptoms": "Small pinpoint brown lesions expanding into circular spots with light brown centers and dark concentric rings; lesions on fruit are circular, sunken, with dark brown centers.",
        "treatment": [
            "Apply protective fungicides (Chlorothalonil, Mancozeb) or strobilurins (Azoxystrobin, Pyraclostrobin) at first lesion identification.",
            "Prune infected foliage and improve internal canopy aeration.",
            "Ensure fungicide sprays achieve thorough under-canopy leaf surface coverage.",
            "Remove and destroy infected plant debris at the end of the harvest season."
        ],
        "prevention": [
            "Maintain wide plant spacing (60-90 cm) on raised plastic-mulched beds.",
            "Avoid overhead irrigation; schedule drip irrigation early in the day.",
            "Rotate with non-host crops like sweet corn or sorghum for at least 2 seasons.",
            "Sanitize trellising stakes and greenhouse structures prior to replanting."
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "crop": "Tomato",
        "pathogen": "Virus (Begomovirus - TYLCV, transmitted by Whiteflies)",
        "severity": "Critical",
        "description": "A devastating plant virus transmitted persistently by the sweetpotato/silverleaf whitefly (Bemisia tabaci). Infected young plants become severely stunted with upward-cupped leaves and drop nearly all flowers, resulting in total yield loss.",
        "symptoms": "Severe upward rolling and curling of leaflet margins, pronounced interveinal chlorosis and yellowing, bush-like stunted plant architecture, and flower abortion.",
        "treatment": [
            "No chemical cure exists for viral infection; immediately rogue out, bag, and destroy infected plants to protect remaining crop.",
            "Control whitefly insect vectors with systemic insecticides (Dinotefuran, Acetamiprid, Spiromesifen) or insecticidal soaps.",
            "Deploy yellow sticky cards (1 card per 20 sq meters) to monitor and trap adult whitefly populations.",
            "Apply reflective silver or metallized plastic mulches to deter incoming whiteflies from landing on plants."
        ],
        "prevention": [
            "Plant TYLCV-resistant tomato hybrids (varieties carrying the Ty-1, Ty-2, or Ty-3 resistance genes).",
            "Use fine 50-mesh insect-proof screening on greenhouse vents and high-tunnel openings.",
            "Maintain a 2-month host-free period between tomato cropping cycles.",
            "Eradicate broadleaf weed hosts (nightshades, mallows, jimsonweed) surrounding the perimeter of the field."
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Tobamovirus (ToMV)",
        "crop": "Tomato",
        "pathogen": "Virus (Tobamovirus - ToMV)",
        "severity": "Critical",
        "description": "An extraordinarily resilient, mechanically transmitted tobamovirus. It can survive in dry plant residue for years, on greenhouse tools, clothing, and seed coats. Spread effortlessly by hands, shears, and plant-to-plant contact.",
        "symptoms": "Light and dark green mosaic or mottled leaf pattern, blistered or puckered leaf surfaces, 'shoestring' narrowing of leaves (fern-like distortion), stunted growth, and internal browning of fruit.",
        "treatment": [
            "No chemical treatment exists; immediately uproot and discard infected plants in sealed bags.",
            "Wash hands thoroughly with soap and water or nonfat dry milk (20% solution) before handling tomato plants.",
            "Disinfect all cutting tools and pruning equipment in a 20% nonfat dry milk solution or 10% trisodium phosphate (TSP).",
            "Enforce a strict no-smoking/no-tobacco policy in tomato greenhouses (tobacco products carry mosaic virus particles)."
        ],
        "prevention": [
            "Plant resistant tomato varieties (look for 'ToMV' or 'TMV' resistance ratings on seed packets).",
            "Purchase certified virus-free seed treated with trisodium phosphate or hot water.",
            "Steam-sterilize or discard contaminated potting soil and seedling trays before reuse.",
            "Rotate planting beds away from solanaceous species for at least 3 years."
        ],
    },
    "Tomato___healthy": {
        "title": "Healthy Tomato Foliage (Solanum lycopersicum)",
        "crop": "Tomato",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "Tomato foliage shows robust, glandular, compound dark green leaflets, strong trichome pubescence, upright vigorous branching, and active flower cluster development without spotting, curling, or chlorosis.",
        "symptoms": "No visible pathology. Intense deep green color, intact serrated leaflets, and healthy aromatic foliage.",
        "treatment": [
            "No curative intervention required.",
            "Provide consistent soil moisture (2.5-3.5 cm water/week) via base drip lines to ensure steady calcium uptake.",
            "Support plants with sturdy cages or Florida-weave trellising to elevate vines."
        ],
        "prevention": [
            "Prune bottom suckers up to the first flower cluster to encourage airflow and prevent soil contact.",
            "Apply straw, compost, or plastic mulch around the root zone.",
            "Provide balanced fertilizer with adequate calcium to prevent blossom end rot and leaf tip burn."
        ],
    },

    # ---------------- GENERIC ALIASES & FALLBACKS ----------------
    "healthy": {
        "title": "Healthy Leaf",
        "crop": "General",
        "pathogen": "None (Healthy Tissue)",
        "severity": "Low",
        "description": "The leaf is classified as healthy and uninfected. Foliage demonstrates optimal chlorophyll concentration, robust cuticle integrity, and normal cell turgor.",
        "symptoms": "No disease symptoms or necrotic lesions observed.",
        "treatment": [
            "Continue standard watering, sunlight, and crop care routines.",
            "Inspect leaves regularly for unusual spots or discoloration.",
            "Maintain suitable soil drainage and plant spacing."
        ],
        "prevention": [
            "Maintain balanced soil nutrition and organic compost amendments.",
            "Employ drip irrigation to keep foliar surfaces dry.",
            "Sanitize tools and remove fallen plant debris."
        ],
    },
    "apple_scab": {
        "title": "Apple Scab",
        "crop": "Apple",
        "pathogen": "Fungus (Venturia inaequalis)",
        "severity": "Moderate",
        "description": "A fungal disease affecting apple leaves and fruit, causing olive-green velvety spots and defoliation.",
        "symptoms": "Olive-green to dark spots may develop on leaves.",
        "treatment": [
            "Prune infected twigs and branches during dry weather.",
            "Apply bio-fungicides such as Copper Octanoate or Liquid Copper at green tip through petal fall.",
            "Collect and incinerate fallen leaves to destroy overwintering fungal pseudothecia."
        ],
        "prevention": [
            "Plant scab-resistant cultivars such as Liberty or Enterprise.",
            "Ensure good tree spacing and prune canopy to maximize airflow.",
            "Water with ground drip emitters rather than overhead sprinklers."
        ],
    },
    "black_rot": {
        "title": "Black Rot",
        "crop": "Fruit Crops",
        "pathogen": "Fungus (Botryosphaeria / Guignardia)",
        "severity": "High",
        "description": "A destructive fungal disease affecting apples and grapes, causing foliar leaf spotting and fruit mummification.",
        "symptoms": "Dark brown circular lesions with purple borders, fruit turning into black shriveled mummies.",
        "treatment": [
            "Prune out dead blighted twigs, cankers, and mummified fruit.",
            "Disinfect pruning tools with 70% alcohol between cuts.",
            "Apply protective fungicides containing captan or copper starting at early cluster development."
        ],
        "prevention": [
            "Prune canopy annually to reduce interior humidity.",
            "Remove mummies and pruned wood from orchard perimeter.",
            "Protect trunks from mechanical lacerations and winter injury."
        ],
    },
    "cedar_rust": {
        "title": "Cedar Apple Rust",
        "crop": "Apple",
        "pathogen": "Fungus (Gymnosporangium juniperi-virginianae)",
        "severity": "Moderate",
        "description": "A fungal rust requiring both apple trees and Eastern red cedar to complete its life cycle.",
        "symptoms": "Yellow-orange or fiery reddish-orange circular spots on apple leaves, developing aecial tubes underneath.",
        "treatment": [
            "Apply protective fungicides (Myclobutanil, Mancozeb) starting at pink bud through petal fall.",
            "Pick off and bag isolated rusted leaves early in the season.",
            "Apply curative systemic triazole sprays within 48 hours of suspected spring rain spore events."
        ],
        "prevention": [
            "Remove nearby cedar galls and Eastern red cedar trees where feasible.",
            "Plant resistant apple varieties like Redfree, Liberty, or Freedom.",
            "Space orchard rows in line with prevailing winds for rapid drying."
        ],
    },
    "powdery_mildew": {
        "title": "Powdery Mildew",
        "crop": "General Crops",
        "pathogen": "Fungus (Erysiphales family)",
        "severity": "Moderate",
        "description": "A common fungal infection affecting cherries, apples, squash, and ornamentals.",
        "symptoms": "White or grayish powdery fungal patches on upper or lower leaf surfaces.",
        "treatment": [
            "Apply potassium bicarbonate (Kaligreen, MilStop) or neem oil at first sign of white spots.",
            "Apply bio-fungicides containing Bacillus amyloliquefaciens every 7-10 days.",
            "Prune out heavily infested shoots and suckers."
        ],
        "prevention": [
            "Select powdery mildew-resistant plant cultivars.",
            "Space plants adequately to facilitate maximum airflow and sunlight.",
            "Use drip irrigation to keep foliar surfaces dry."
        ],
    },
    "cercospora_gray_leaf_spot": {
        "title": "Cercospora / Gray Leaf Spot",
        "crop": "Corn",
        "pathogen": "Fungus (Cercospora zeae-maydis)",
        "severity": "High",
        "description": "A serious foliar fungal disease causing rectangular, vein-delimited lesions on corn leaves.",
        "symptoms": "Rectangular tan-to-gray lesions running parallel with leaf veins.",
        "treatment": [
            "Apply strobilurin or triazole fungicides at VT-R1 if lesions appear on ear leaves.",
            "Ensure thorough spray coverage of the upper ear leaves.",
            "Harvest early if stalk lodging or canopy dieback is severe."
        ],
        "prevention": [
            "Plant corn hybrids with high Cercospora genetic tolerance.",
            "Implement a 2-year crop rotation with non-host crops like soybeans.",
            "Utilize conservation tillage to accelerate corn stover decomposition."
        ],
    },
    "common_rust": {
        "title": "Common Rust",
        "crop": "Corn",
        "pathogen": "Fungus (Puccinia sorghi)",
        "severity": "Moderate",
        "description": "Foliar rust disease causing cinnamon-brown powdery pustules on both leaf surfaces.",
        "symptoms": "Cinnamon-brown powdery pustules erupting across upper and lower leaf surfaces.",
        "treatment": [
            "Apply foliar fungicides (Pyraclostrobin, Azoxystrobin) if pustules exceed threshold before silking.",
            "Repeat application within 14 days if damp overcast conditions continue."
        ],
        "prevention": [
            "Plant resistant corn hybrids containing Rp1-D resistance traits.",
            "Plant early to complete grain fill before peak late-summer spore arrival.",
            "Maintain balanced soil potassium and nitrogen."
        ],
    },
    "northern_leaf_blight": {
        "title": "Northern Leaf Blight",
        "crop": "Corn",
        "pathogen": "Fungus (Exserohilum turcicum)",
        "severity": "High",
        "description": "A severe fungal blight causing large cigar-shaped grayish lesions on corn leaves.",
        "symptoms": "Long, cigar-shaped grayish-green to tan lesions (3-15 cm) with smooth margins.",
        "treatment": [
            "Apply dual-action fungicides (QoI + DMI) when lesions reach threshold levels on ear leaves.",
            "Use high-clearance sprayers with drop nozzles for optimal ear zone coverage."
        ],
        "prevention": [
            "Select corn hybrids with Ht multi-gene resistance.",
            "Rotate fields out of corn for at least 1-2 seasons.",
            "Incorporate corn residue with tillage to accelerate microbial breakdown."
        ],
    },
    "bacterial_spot": {
        "title": "Bacterial Spot",
        "crop": "Tomato / Pepper / Peach",
        "pathogen": "Bacterium (Xanthomonas spp.)",
        "severity": "High",
        "description": "A bacterial pathogen causing small dark spots, leaf yellowing, and severe defoliation.",
        "symptoms": "Small, dark water-soaked spots on leaves and scabby raised spots on fruit.",
        "treatment": [
            "Apply copper bactericide tank-mixed with Mancozeb at 5-7 day intervals.",
            "Apply bacteriophage biological sprays (AgriPhage) during active outbreaks.",
            "Disinfect pruning tools and stakes with 70% alcohol or 10% bleach."
        ],
        "prevention": [
            "Plant certified disease-free, hot-water-treated seed and resistant varieties.",
            "Avoid handling plants when foliage is wet.",
            "Use drip irrigation exclusively; avoid overhead watering.",
            "Enforce a 3-year crop rotation away from solanaceous plants."
        ],
    },
    "early_blight": {
        "title": "Early Blight",
        "crop": "Potato / Tomato",
        "pathogen": "Fungus (Alternaria solani)",
        "severity": "Moderate",
        "description": "A fungal disease causing dark brown concentric target spots on mature foliage.",
        "symptoms": "Dark brown spots with concentric target rings and surrounding yellowing.",
        "treatment": [
            "Remove and destroy infected lower leaves up to 45 cm above soil.",
            "Apply protective fungicides (Chlorothalonil, Mancozeb, Copper) at 7-10 day intervals.",
            "Prune lower suckers to improve ventilation."
        ],
        "prevention": [
            "Apply clean straw or plastic mulch to stop soil spores from splashing.",
            "Stake or cage plants to elevate foliage off wet soil.",
            "Water only with drip irrigation at ground level.",
            "Rotate crops for 3-4 years away from solanaceous crops."
        ],
    },
    "late_blight": {
        "title": "Late Blight",
        "crop": "Potato / Tomato",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "severity": "Critical",
        "description": "A destructive water-mold causing rapid foliar collapse and fruit/tuber rot during cool, damp weather.",
        "symptoms": "Water-soaked dark lesions expanding rapidly, white downy fungal bloom under leaves.",
        "treatment": [
            "Apply curative systemic oomycides (Revus, Curzate) immediately upon detection.",
            "Apply high-dose protective sprays (Chlorothalonil, Mancozeb) on surrounding plants.",
            "Remove and destroy heavily blighted plants immediately; bag on-site."
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes or resistant tomato hybrids (Ph-2/Ph-3).",
            "Eliminate all potato cull piles and volunteer plants.",
            "Hill potatoes generously to protect tubers from surface spore wash-off.",
            "Kill potato vines 2 weeks prior to tuber digging."
        ],
    },
    "leaf_mold": {
        "title": "Leaf Mold",
        "crop": "Tomato",
        "pathogen": "Fungus (Passalora fulva)",
        "severity": "Moderate",
        "description": "A fungal disease thriving in high-humidity high tunnels and greenhouses.",
        "symptoms": "Yellow patches on upper leaf surfaces, olive-brown velvety spore mold underneath.",
        "treatment": [
            "Prune infected lower foliage immediately.",
            "Apply bio-fungicides (Bacillus subtilis, copper soap) weekly.",
            "Ventilate and run circulation fans in high tunnels to drop humidity below 80%."
        ],
        "prevention": [
            "Select resistant tomato cultivars (Cf gene resistance).",
            "Space plants widely and prune suckers to optimize air movement.",
            "Avoid late evening watering; keep foliage completely dry overnight."
        ],
    },
    "septoria_leaf_spot": {
        "title": "Septoria Leaf Spot",
        "crop": "Tomato",
        "pathogen": "Fungus (Septoria lycopersici)",
        "severity": "Moderate",
        "description": "A severe foliar disease causing numerous small spots with gray centers and black specks.",
        "symptoms": "Circular spots with dark borders and grayish centers containing tiny black pycnidia.",
        "treatment": [
            "Prune off infected lower leaves as soon as spots appear.",
            "Apply protective fungicides (Chlorothalonil, Mancozeb, Copper) on a 7-day schedule.",
            "Collect and dispose of fallen diseased leaves."
        ],
        "prevention": [
            "Mulch beds with straw or plastic to prevent soil splash.",
            "Eradicate solanaceous weeds around the garden.",
            "Use drip irrigation to keep foliage dry.",
            "Enforce a 3-year crop rotation."
        ],
    },
    "spider_mites": {
        "title": "Spider Mites",
        "crop": "Tomato / Vegetables",
        "pathogen": "Arachnid Pest (Tetranychus urticae)",
        "severity": "Moderate",
        "description": "Tiny arachnids causing foliar stippling, chlorosis, and fine silken webs in hot dry weather.",
        "symptoms": "Yellow stippling on leaves, fine silken webbing under leaves, leaves bronzing and dropping.",
        "treatment": [
            "Spray plants with strong water blast to dislodge mites and webs.",
            "Apply insecticidal soap or horticultural neem oil thoroughly coating leaf undersides.",
            "Release predatory mites (Phytoseiulus persimilis) in greenhouse environments."
        ],
        "prevention": [
            "Suppress dust around garden pathways with mulch or watering.",
            "Avoid broad-spectrum pyrethroid insecticides that kill beneficial predator insects.",
            "Maintain optimal plant watering; drought-stressed plants are preferred by mites."
        ],
    },
    "target_spot": {
        "title": "Target Spot",
        "crop": "Tomato",
        "pathogen": "Fungus (Corynespora cassiicola)",
        "severity": "Moderate",
        "description": "A warm-climate fungal pathogen causing brown concentric lesions on leaves and fruit.",
        "symptoms": "Pinpoint brown spots expanding into target-like concentric rings, sunken fruit lesions.",
        "treatment": [
            "Apply fungicides (Chlorothalonil, Mancozeb, Azoxystrobin) at first lesion sign.",
            "Prune lower infected foliage to open canopy air pathways."
        ],
        "prevention": [
            "Space plants 60-90 cm apart on raised plastic-mulched beds.",
            "Use drip irrigation early in the day.",
            "Rotate with non-host crops like corn or sorghum for 2 seasons."
        ],
    },
    "yellow_leaf_curl_virus": {
        "title": "Tomato Yellow Leaf Curl Virus",
        "crop": "Tomato",
        "pathogen": "Virus (Begomovirus, Whitefly-vectored)",
        "severity": "Critical",
        "description": "A devastating virus transmitted by whiteflies, causing severe stunting and leaf cupping.",
        "symptoms": "Upward curled leaflet margins, yellowing between veins, bushy stunted plants, blossom drop.",
        "treatment": [
            "Rogue out and destroy infected plants in sealed bags immediately.",
            "Control whitefly insect vectors with insecticidal soaps or systemic treatments.",
            "Deploy yellow sticky cards to trap adult whiteflies."
        ],
        "prevention": [
            "Plant TYLCV-resistant tomato hybrids (Ty gene resistance).",
            "Use 50-mesh insect-proof screening on high tunnels and greenhouses.",
            "Apply reflective silver mulches to repel whiteflies.",
            "Eliminate broadleaf weeds around field margins."
        ],
    },
    "mosaic_virus": {
        "title": "Mosaic Virus",
        "crop": "Tomato / Pepper",
        "pathogen": "Virus (Tobamovirus - ToMV / TMV)",
        "severity": "Critical",
        "description": "A highly stable, mechanically transmitted virus causing mottled foliage and stunted growth.",
        "symptoms": "Mottled light and dark green mosaic patterns, leaf blistering, and distorted shoestring leaves.",
        "treatment": [
            "Remove and destroy infected plants immediately.",
            "Wash hands and tools with 20% nonfat dry milk or 10% TSP solution.",
            "Prohibit tobacco use near plants (tobacco can harbor mosaic virus)."
        ],
        "prevention": [
            "Plant resistant tomato cultivars (ToMV/TMV labeled).",
            "Use certified virus-free seed treated with trisodium phosphate.",
            "Disinfect greenhouse benches, pots, and stakes before reuse."
        ],
    },
    "citrus_greening": {
        "title": "Citrus Greening (Huanglongbing)",
        "crop": "Citrus",
        "pathogen": "Bacterium (Candidatus Liberibacter)",
        "severity": "Critical",
        "description": "A fatal bacterial disease transmitted by psyllids, causing blotchy yellowing and bitter fruit.",
        "symptoms": "Asymmetrical blotchy yellow mottle on leaves, small misshapen bitter fruit, tree decline.",
        "treatment": [
            "Remove and destroy confirmed infected trees to protect surrounding groves.",
            "Control Asian citrus psyllid vectors with targeted treatments.",
            "Support declining trees with foliar micro-nutrient sprays (Zn, Mn, Fe)."
        ],
        "prevention": [
            "Plant certified pathogen-free nursery stock grown in screenhouses.",
            "Monitor yellow sticky traps for citrus psyllid detection.",
            "Comply with citrus quarantine movement regulations."
        ],
    },
    "esca": {
        "title": "Grape Esca (Black Measles)",
        "crop": "Grape",
        "pathogen": "Fungal Complex",
        "severity": "Critical",
        "description": "A vascular wood disease of grapevines causing tiger-stripe foliar necrosis and fruit spots.",
        "symptoms": "Tiger-stripe interveinal leaf scorch, dark purple spots on fruit, vine dieback.",
        "treatment": [
            "Surgically cut trunks below infected rot columns to train renewal suckers.",
            "Mark symptomatic vines and prune them last during dormancy.",
            "Seal large pruning wounds with pruning paint or Trichoderma paste."
        ],
        "prevention": [
            "Practice double pruning or late winter pruning to avoid peak spore release.",
            "Sanitize shears between vines with 70% alcohol.",
            "Plant certified disease-tested rootstocks and scions."
        ],
    },
    "grape_leaf_blight": {
        "title": "Grape Leaf Blight",
        "crop": "Grape",
        "pathogen": "Fungus (Pseudocercospora vitis)",
        "severity": "Moderate",
        "description": "Late-season foliar blight causing dark brown angular spots and premature defoliation.",
        "symptoms": "Angular dark reddish-brown to black spots on leaves, coalescing into blighted zones.",
        "treatment": [
            "Apply copper-based fungicides or Mancozeb at first symptom appearance.",
            "Maintain post-harvest protective sprays if wet weather continues.",
            "Gather and destroy fallen blighted leaves."
        ],
        "prevention": [
            "Maintain open canopy architecture to reduce humidity.",
            "Avoid excessive late-season nitrogen fertilization.",
            "Use drip irrigation beneath the canopy."
        ],
    },
    "leaf_scorch": {
        "title": "Leaf Scorch",
        "crop": "Strawberry",
        "pathogen": "Fungus (Diplocarpon earlianum)",
        "severity": "Moderate",
        "description": "Fungal disease causing numerous dark purple spots that coalesce, making leaves look scorched.",
        "symptoms": "Numerous small dark purple to brown spots, leaves curling and turning brown.",
        "treatment": [
            "Apply protective fungicides (Captan, Thiram) starting at early spring crown growth.",
            "Prune off severely scorched leaves during bed renovation.",
            "Mow or renovate strawberry beds immediately after harvest."
        ],
        "prevention": [
            "Plant certified disease-free strawberry crowns.",
            "Choose resistant varieties like Allstar or Honeoye.",
            "Space plants for rapid drying and use drip irrigation under mulch."
        ],
    },
    "unknown": {
        "title": "Unknown Condition",
        "crop": "Unknown",
        "pathogen": "Unidentified / Out-of-Distribution",
        "severity": "Moderate",
        "description": "The foliar image could not be decisively categorized into known taxonomy classes.",
        "symptoms": "The leaf may exhibit atypical chlorosis, mechanical injury, insect feeding, or uncommon nutritional deficiency.",
        "treatment": [
            "Quarantine the specimen or isolate the affected plant if symptoms are rapidly spreading.",
            "Avoid applying broad-spectrum pesticides until the underlying cause is properly diagnosed.",
            "Capture multiple well-lit, sharp macro photographs of the leaf upper and lower surfaces.",
            "Submit samples to your local university agricultural extension office or botanical lab."
        ],
        "prevention": [
            "Ensure standard baseline cultural hygiene: sterilize pruning shears in 70% isopropyl alcohol.",
            "Switch strictly to drip or root-zone watering to prevent foliar disease spread.",
            "Maintain optimal soil pH and test electrical conductivity (EC) to rule out fertilizer burn."
        ],
    },
}

# Automatically ensure every entry has a unified 'care' list (treatment + prevention)
for key, data in list(DISEASE_INFO.items()):
    treat = data.get("treatment", [])
    prev = data.get("prevention", [])
    if "care" not in data or not data["care"]:
        data["care"] = treat + prev
    elif "treatment" not in data:
        data["treatment"] = data["care"][:2]
        data["prevention"] = data["care"][2:]


def get_disease_info(label: str) -> dict[str, Any]:
    """
    Match a model's PlantVillage label to rich educational, treatment, and prevention data.
    First checks exact 38-class match, then pattern matches by condition and crop.
    """
    raw_label = str(label or "").strip()

    # 1. Exact match in 38-class dictionary
    if raw_label in DISEASE_INFO:
        return DISEASE_INFO[raw_label].copy()

    # 2. Case-insensitive exact match
    for k in DISEASE_INFO:
        if k.lower() == raw_label.lower():
            return DISEASE_INFO[k].copy()

    # 3. Normalized matching
    if "___" in raw_label:
        crop_raw, condition_raw = raw_label.split("___", 1)
    else:
        crop_raw = ""
        condition_raw = raw_label

    crop_name = re.sub(r"_+", " ", crop_raw).strip(" ,_")
    condition = re.sub(r"[^a-z0-9]+", "_", condition_raw.lower()).strip("_")

    key = "unknown"
    if "healthy" in condition:
        # Match crop-specific healthy if available
        matched = False
        for k in DISEASE_INFO:
            if "healthy" in k.lower() and crop_name.lower() in k.lower():
                key = k
                matched = True
                break
        if not matched:
            key = "healthy"
    elif "apple_scab" in condition:
        key = "Apple___Apple_scab"
    elif "cedar_apple_rust" in condition or "cedar_rust" in condition:
        key = "Apple___Cedar_apple_rust"
    elif "black_rot" in condition:
        if "grape" in crop_name.lower():
            key = "Grape___Black_rot"
        else:
            key = "Apple___Black_rot"
    elif "powdery_mildew" in condition:
        if "cherry" in crop_name.lower():
            key = "Cherry_(including_sour)___Powdery_mildew"
        elif "squash" in crop_name.lower():
            key = "Squash___Powdery_mildew"
        else:
            key = "powdery_mildew"
    elif "cercospora" in condition or "gray_leaf_spot" in condition:
        key = "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    elif "common_rust" in condition:
        key = "Corn_(maize)___Common_rust_"
    elif "northern_leaf_blight" in condition:
        key = "Corn_(maize)___Northern_Leaf_Blight"
    elif "bacterial_spot" in condition:
        if "peach" in crop_name.lower():
            key = "Peach___Bacterial_spot"
        elif "pepper" in crop_name.lower():
            key = "Pepper,_bell___Bacterial_spot"
        elif "tomato" in crop_name.lower():
            key = "Tomato___Bacterial_spot"
        else:
            key = "bacterial_spot"
    elif "early_blight" in condition:
        if "potato" in crop_name.lower():
            key = "Potato___Early_blight"
        else:
            key = "Tomato___Early_blight"
    elif "late_blight" in condition:
        if "potato" in crop_name.lower():
            key = "Potato___Late_blight"
        else:
            key = "Tomato___Late_blight"
    elif "leaf_mold" in condition:
        key = "Tomato___Leaf_Mold"
    elif "septoria_leaf_spot" in condition:
        key = "Tomato___Septoria_leaf_spot"
    elif "spider_mite" in condition:
        key = "Tomato___Spider_mites Two-spotted_spider_mite"
    elif "target_spot" in condition:
        key = "Tomato___Target_Spot"
    elif "yellow_leaf_curl_virus" in condition or "curl" in condition:
        key = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    elif "mosaic_virus" in condition:
        key = "Tomato___Tomato_mosaic_virus"
    elif "haunglongbing" in condition or "citrus_greening" in condition:
        key = "Orange___Haunglongbing_(Citrus_greening)"
    elif "esca" in condition or "black_measles" in condition:
        key = "Grape___Esca_(Black_Measles)"
    elif "isariopsis" in condition or "leaf_blight" in condition:
        key = "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
    elif "leaf_scorch" in condition:
        key = "Strawberry___Leaf_scorch"
    else:
        # Fallback to key lookup
        for k in DISEASE_INFO:
            if condition in k.lower() or k.lower() in condition:
                key = k
                break

    if key not in DISEASE_INFO:
        key = "unknown"

    info = DISEASE_INFO[key].copy()
    if crop_name and "___" not in key and key != "unknown":
        if key == "healthy":
            info["title"] = f"Healthy Leaf — {crop_name}"
        else:
            info["title"] = f"{crop_name}: {info['title']}"

    return info


_APP_ROOT = Path(__file__).resolve().parent
_THUMBNAILS_DIR = _APP_ROOT / "assets" / "disease_thumbnails"

@functools.lru_cache(maxsize=128)
def get_disease_thumbnail_base64(key: str) -> str:
    """
    Return base64 encoded data URI of the actual leaf specimen thumbnail for the given disease key.
    Resolves directly from assets/disease_thumbnails with fallback to healthy leaf.
    """
    clean = (key or "").strip().lower()
    
    # 1. Try exact filename match
    img_file = _THUMBNAILS_DIR / f"{clean}.jpg"
    if img_file.exists():
        try:
            b64_str = base64.b64encode(img_file.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception:
            pass

    # 2. Try normalized name (replacing spaces, parens)
    norm = re.sub(r"[^a-z0-9_]+", "_", clean).strip("_")
    img_file = _THUMBNAILS_DIR / f"{norm}.jpg"
    if img_file.exists():
        try:
            b64_str = base64.b64encode(img_file.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception:
            pass

    # 3. Match by partial or key containment in existing thumbnail files
    if _THUMBNAILS_DIR.exists():
        for thumb in _THUMBNAILS_DIR.glob("*.jpg"):
            stem = thumb.stem.lower()
            if stem in clean or clean in stem:
                try:
                    b64_str = base64.b64encode(thumb.read_bytes()).decode("ascii")
                    return f"data:image/jpeg;base64,{b64_str}"
                except Exception:
                    pass

    # 4. Fallback to healthy.jpg
    fallback = _THUMBNAILS_DIR / "healthy.jpg"
    if fallback.exists():
        try:
            b64_str = base64.b64encode(fallback.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception:
            pass

    return ""
