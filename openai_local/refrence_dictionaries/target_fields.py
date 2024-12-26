moroccan_regions = [
    "Tanger-Tetouan-Al Hoceima",
    "Oriental",
    "Fes-Meknes",
    "Rabat-Sale-Kenitra",
    "Beni Mellal-Khenifra",
    "Casablanca-Settat",
    "Marrakech-Safi",
    "Draa-Tafilalet",
    "Souss-Massa",
    "Guelmim-Oued Noun",
    "Laayoune-Sakia El Hamra",
    "Dakhla-Oued Ed-Dahab"
]
class FieldPrompts:
        Désignation_Organisme_Expropriant_prompt = ( "ما هي تسمية الجهة المعنية بالنزع؟")
        Objet_prompt = ("وصف الغرض أو الهدف المحدد من مشروع نزع الملكية. ركز على هدف المشروع دون إضافة سياق إضافي.")
        Jugement_Ref_prompt = ("إذا ظهرت مصطلحات 'حكم' أو 'مقرر حكم'، استخرج المصطلحات وقدم مرجع الحكم الرسمي. إذا لم يتم العثور على هذه المصطلحات، قم بالرد بـ 'NA'. قم بذكر المرجع فقط أو 'NA'، وفي حال عدم وجود مرجع حكم، أجب بـ 'NA'.")
        Date_Jugement_prompt = ("اذكر تاريخ الحكم بصيغة DD/MM/YYYYYY إذا تم ذكر ’حكم‘ أو ’مقرر حكم‘، وإذا لم يتم ذكره، أجب بـ ’NA‘. قم بتضمين التاريخ فقط أو 'NA'.إذا لم يتم ذكر التاريخ، أجب بـ 'NA'.")
        Num_Décision_prompt = ("استخرج رقم القرار الرسمي واذكر رقم القرار الرسمي إذا كان مصطلح 'قرار' موجودًا في النص، وإذا لم يكن مذكورًا، أجب بـ 'لا يوجد'. أدرج الرقم فقط أو 'NA'.إذا لم يكن هناك رقم قرار، أجب بـ 'NA'.")
        Num_Bulletin_Officiel_prompt = ("إذا ذكرت ”الجريدة الرسمية“، اذكر رقم الجريدة الرسمية، إذا لم يتم ذكرها، أجب بـ ”NA“. أدرج الرقم فقط أو 'NA'، إذا لم يكن هناك ذكر، أجب بـ 'NA'.")
        Date_Bulletin_Officiel_prompt = ("اذكر تاريخ النشر في الجريدة الرسمية بصيغة DD/MM/YYYYY إذا تم ذكر 'الجريدة الرسمية' وإذا لم يتم ذكره، أجب بـ 'NA'. قم بتضمين التاريخ فقط أو 'NA'.إذا لم يتم ذكر التاريخ، أجب بـ 'NA'.")
        Référence_Loi_prompt = ("اذكر الرقم المرجعي القانوني المرتبط بقانون نزع الملكية. أجب بالرقم المرجعي فقط، دون أي سياق إضافي.")
        Référence_Décret_Expropriation_prompt = ("قم باستخراج وتقديم رقم المرسوم بصيغة 'x.xx.xxx' إذا تم ذكر 'مرسوم'، قم بتضمين الرقم المرجعي فقط، إذا تم العثور على أكثر من مرسوم اذكرهم جميعاً")
        Région_frappée_par_le_projet_prompt = (
            f"Identify and specify the Moroccan region impacted by the project from the following list: {moroccan_regions}. "
            f"Provide only the region name from the list.\n"
        )
        Commune_frappée_par_le_projet_prompt = (
            "حدد البلدية  المتأثرة بالمشروع، اذكر اسم البلدية فقط."
        )
        Intitulé_Projet_prompt = (
            "اذكر العنوان الرسمي لمشروع نزع الملكية"
        )
        Tranche_Projet_d_expropriation_prompt = (
            "Indicate the phase or tranche of the expropriation project. "
            "Provide only the tranche details.\n"
        )
        Montant_Total_Projet_prompt = (
            "What's teh Highest amount in the Document  "
            "Provide only the amount without additional details.\n"
        )
        Montant_Global_consigné_prompt = (
            "Provide the total amount of funds deposited for the expropriation in Moroccan Dirhams. "
            "Include only the monetary amount.\n"

        )

        Type_Versement_prompt = (
            "Specify the payment method used for the expropriation process. "
            "Provide only the payment type.\n"

        )

        Nom_Prénom_prompt = (
            "List all names found in the text, presenting them in a bullet-point format. "
            "Provide names only, without extra text.\n"
        )

        Adresse_courrier_prompt = (
            "Provide the full mailing address of the expropriating organization. "
            "Include only the address.\n"

        )

        Num_Téléphone_prompt = (
            "Provide the organization's phone number, including the international code if applicable. "
            "Include only the number.\n"
        )

        Localité_prompt = (
            "State the locality or area where the project is situated. "
            "Provide only the name of the locality.\n"

        )

        Identifiant_Fiscal_prompt = (
            "Extract and provide the fiscal identifier (IF) of any companies involved, if available. "
            "Provide only the identifier numbers.\n"

        )

# Fields Prompts
# class FieldPrompts:
#     Désignation_Organisme_Expropriant_prompt = (
#         "Identify and state the full official name of the organization responsible for the expropriation. "
#         "Ensure the response is concise and includes only the organization's name.\n"
#     )
#
#     Objet_prompt = (
#         "Describe the specific purpose or objective of the expropriation project. "
#         "Focus on the project's aim without adding extra context.\n"
#     )
#
#     Jugement_Ref_prompt = (
#         "If the terms 'حكم' or 'مقرر حكم' appear, extract and provide the official judgment reference. "
#         "If these terms are not found, reply with 'NA'. Include only the reference or 'NA'.\n"
#         "If there is no judgment reference, respond: NA."
#     )
#
#     Date_Jugement_prompt = (
#         "Provide the date of the judgment in DD/MM/YYYY format if 'حكم' or 'مقرر حكم' is mentioned. "
#         "If not mentioned, respond with 'NA'. Include only the date or 'NA'.\n"
#         "If no date is provided, respond: NA."
#     )
#
#     Num_Décision_prompt = (
#         "Extract and state the official decision number if the term 'قرار' is present in the text. "
#         "If not mentioned, reply with 'NA'. Include only the number or 'NA'.\n"
#         "If there is no decision number, respond: NA."
#     )
#
#     Num_Bulletin_Officiel_prompt = (
#         "If 'الجريدة الرسمية' is mentioned, provide the number of the official bulletin. "
#         "If not mentioned, respond with 'NA'. Include only the number or 'NA'.\n"
#         "If there is no mention, respond: NA."
#     )
#
#     Date_Bulletin_Officiel_prompt = (
#         "State the date of the publication in the official bulletin in DD/MM/YYYY format if 'الجريدة الرسمية' is mentioned. "
#         "If not mentioned, respond with 'NA'. Include only the date or 'NA'.\n"
#         "If no date is given, respond: NA."
#     )
#
#     Référence_Loi_prompt = (
#         "Provide the legal reference number associated with the expropriation law. "
#         "Answer with the reference number only, without additional context.\n"
#     )
#
#     Référence_Décret_Expropriation_prompt = (
#         "Extract and provide the decree reference in the format 'x.xx.xxx' if 'مرسوم' is mentioned. "
#         "Include only the reference number.\n"
#         "If more than one is found list all of them"
#     )
#
#     Région_frappée_par_le_projet_prompt = (
#         f"Identify and specify the Moroccan region impacted by the project from the following list: {moroccan_regions}. "
#         f"Provide only the region name from the list.\n"
#     )
#
#     Commune_frappée_par_le_projet_prompt = (
#         "Identify and name the specific commune affected by the project. "
#         "Provide only the commune name.\n"
#     )
#
#     Intitulé_Projet_prompt = (
#         "State the official title of the expropriation project. "
#         "Include only the project title as it is officially recognized.\n"
#     )
#
#     Tranche_Projet_d_expropriation_prompt = (
#         "Indicate the phase or tranche of the expropriation project. "
#         "Provide only the tranche details.\n"
#     )
#
#     Montant_Total_Projet_prompt = (
#         "What's teh Highest amount in the Document  "
#         "Provide only the amount without additional details.\n"
#     )
#
#     Montant_Global_consigné_prompt = (
#         "Provide the total amount of funds deposited for the expropriation in Moroccan Dirhams. "
#         "Include only the monetary amount.\n"
#
#     )
#
#     Type_Versement_prompt = (
#         "Specify the payment method used for the expropriation process. "
#         "Provide only the payment type.\n"
#
#     )
#
#     Nom_Prénom_prompt = (
#         "List all names found in the text, presenting them in a bullet-point format. "
#         "Provide names only, without extra text.\n"
#     )
#
#     Adresse_courrier_prompt = (
#         "Provide the full mailing address of the expropriating organization. "
#         "Include only the address.\n"
#
#     )
#
#     Num_Téléphone_prompt = (
#         "Provide the organization's phone number, including the international code if applicable. "
#         "Include only the number.\n"
#     )
#
#     Localité_prompt = (
#         "State the locality or area where the project is situated. "
#         "Provide only the name of the locality.\n"
#
#     )
#
#     Identifiant_Fiscal_prompt = (
#         "Extract and provide the fiscal identifier (IF) of any companies involved, if available. "
#         "Provide only the identifier numbers.\n"
#
#     )


expropriation_data = [
    {
        "field": "Désignation Organisme Expropriant",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Désignation_Organisme_Expropriant_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "Official name or title of the organization responsible for the expropriation.",
        "regex": r"^[A-Za-z\s]+$"
    },
    {
        "field": "Objet",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Objet_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "Purpose or reason behind the expropriation project.",
        "regex": r"^[A-Za-z\s]+$"
    },
    {
        "field": "Jugement Ref",
        "lookuptext": "table",
        "field_prompt": FieldPrompts.Jugement_Ref_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "Official reference number or ID associated with the judgment.",
        "regex": r"^\d+\/\d+$"
    },
    {
        "field": "Date Jugement",
        "lookuptext": "table",
        "field_prompt": FieldPrompts.Date_Jugement_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The official date on which the judgment was issued.",
        "regex": r"^\d{2}\/\d{2}\/\d{4}$"
    },
    {
        "field": "N° Décision",
        "lookuptext": "table",
        "field_prompt": FieldPrompts.Num_Décision_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The official decision number related to the expropriation process.",
        "regex": r"^\d+$"
    },
    {
        "field": "N° Bulletin Officiel",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Num_Bulletin_Officiel_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The number assigned to the official bulletin where the expropriation information is published.",
        "regex": r"^\d+$"
    },
    {
        "field": "Date Bulletin Officiel",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Date_Bulletin_Officiel_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The date of publication of the official bulletin.",
        "regex": r"^\d{2}\/\d{2}\/\d{4}$"
    },
    {
        "field": "Référence Loi",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Référence_Loi_prompt,
        "dynamic": False,
        "default_value": "7-81",
        "description": "The legal reference governing the expropriation.",
        "regex": r"^\d{1,3}-\d{2}$"
    },
    {
        "field": "Référence Décret Expropriation",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Référence_Décret_Expropriation_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The reference of the decree authorizing the expropriation process.",
        "regex": r"^\d+$"
    },
    {
        "field": "Région frappée par le projet",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Région_frappée_par_le_projet_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The Moroccan region impacted by the expropriation project.",
        "regex": r"^[A-Za-z\s\-]+$"
    },
    {
        "field": "Commune frappée par le projet",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Commune_frappée_par_le_projet_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The specific community impacted by the project.",
        "regex": r"^[A-Za-z\s\-]+$"
    },
    {
        "field": "Intitulé Projet",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Intitulé_Projet_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The official name or title of the expropriation project.",
        "regex": r"^[A-Za-z0-9\s\-]+$"
    },
    {
        "field": "Tranche Projet d'expropriation",
        "lookuptext": "table",
        "field_prompt": FieldPrompts.Tranche_Projet_d_expropriation_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The tranche or phase of the expropriation project.",
        "regex": r"^\d+$"
    },
    {
        "field": "Montant Total Projet",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Montant_Total_Projet_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The total financial cost of the expropriation project.",
        "regex": r"^\d+(\.\d{2})?$"
    },
    {
        "field": "Montant Global consigné",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Montant_Global_consigné_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The total amount of money deposited for the expropriation.",
        "regex": r"^\d+(\.\d{2})?$"
    },
    {
        "field": "Type Versement",
        "lookuptext": "table",
        "field_prompt": FieldPrompts.Type_Versement_prompt,
        "dynamic": True,
        "default_value": "Chéque",
        "description": "The payment method used for the expropriation (e.g., Cheque).",
        "regex": r"^[A-Za-z\s]+$"
    },

    {
        "field": "Localité",
        "lookuptext": "text",
        "field_prompt": FieldPrompts.Localité_prompt,
        "dynamic": True,
        "default_value": "NA",
        "description": "The locality or area where the project is located.",
        "regex": ''
    },

]
