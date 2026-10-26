# Getting the local datasets

The project runs on researcher-supplied files. The full machine-readable inventory is
[config/paths.yaml](../config/paths.yaml): NACC paths are under nacc, the ADNI
dictionary and table selection are under adni, and subject-level table patterns
are under backend. The list below describes the files configured in this
project's local copy. A question loads only the relevant tables; all listed
files are not required for every question.

## Access and placement

### NACC

Request a Quick Access File through the [NACC Data Front Door](https://www.naccdata.org/data-front-door).
NACC's [Researcher's Guide](https://www.naccdata.org/the-nacc-researchers-guide)
explains the request process and secure download link. Follow the terms of your
NACC data use agreement.

For this project's NACC queries, the main file is
NACC/investigator_nacc70.csv: a longitudinal UDS visit table with NACCID,
NACCVNUM, VISITYR, diagnoses, and clinical/cognitive variables. Put your
authorized export at that path, or change nacc.data_csv in config/paths.yaml.
NACC/uds3-rdd.pdf is the UDS3 Researcher's Data Dictionary used to enrich
variable descriptions when building the catalog; set nacc.rdd_pdf if your
download uses another name. The local folder also contains imaging,
neuropathology, genetics, and biomarker RDD supplements, but the current
nacc.rdd_pdf setting does not parse them. The local JMP join file is not an
ingestion source.

### ADNI

Apply for access and accept the data use agreement through
[ADNI's access page](https://adni.loni.usc.edu/data-samples/adni-data/).
After approval, use the LONI IDA **Search and Download → Study Files** view,
as described in the [ADNI Getting Started guide](https://adni.loni.usc.edu/quick-start-guide-asset101625/getting_started.html).
Download the categories in the table below and retain LONI's numbered folder
structure under ADNI/, or change adni.base_dir in config/paths.yaml. The
patterns end in _*.csv so newer date-stamped exports can be selected without
renaming every file. See the [ADNI file tour](https://adni.loni.usc.edu/quick-start-guide-asset/filetour.html)
for category descriptions.

| LONI category | Configured table names | Why they are included |
| --- | --- | --- |
| 1. Start Here / Quick_Start | DATADIC_*.csv | Unified variable dictionary for catalog names, descriptions, and coding. |
| 5. Assessments / Diagnosis and Neuropsychological | DXSUM, BLCHANGE, CDR, MMSE, MOCA, ADAS, FAQ, NEUROBAT | Diagnosis, follow-up dates/status, and cognitive or functional measures. DXSUM and CDR are especially important for diagnosis and follow-up questions. |
| 14. Subject Characteristics and 8. Enrollment | PTDEMOG, FAMHXPAR, FAMHXSIB, ARM, VISITS, ROSTER, REGISTRY | Demographics, family history, participant/visit context, and enrollment. |
| 10. Imaging | UCSFFSX7, MRIMETA, MRI3META, AMYMETA, AV45META, TAUMETA, UCBERKELEY_AMY_6MM, UCBERKELEY_TAU_6MM | MRI/PET availability and derived image measurements. |
| 6. Biospecimen | UPENN_PLASMA_FUJIREBIO_QUANTERIX, UPENNBIOMK_MASTER, UPENNBIOMK_ROCHE_ELECSYS, APOERES, BIOMARK, ALIQUOT_COUNT, LABDATA, LABTESTS | Fluid biomarkers, APOE, specimens, and laboratory metadata. |
| 4. ADSP PHC / ADSP_PHC | ADSP_PHC_COGN, ADSP_PHC_CSF, ADSP_PHC_DEMODX, ADSP_PHC_PLASMA, ADSP_PHC_T1_FS | Harmonized ADNI-linked features. These tables use ADNI identifiers and remain in ID space B. |

This is the complete set of 36 ADNI table patterns configured under
backend.adni_data_files, plus the DATADIC pattern. The local snapshot checked
while writing this guide had one matching file for every pattern, dated
15Jul2026. A new export may have different date suffixes or coverage.
Exact subfolder paths and filename patterns are in config/paths.yaml.
One configuration detail to check before using that harmonized diagnosis table:
ADSP_PHC_DEMODX is listed under backend.adni_data_files but not under
adni.local_tables, so backend reads can find it while catalog ingestion may
omit its dictionary entries.

### Separate ADSP-PHC R3 source

The local backend also has 12 configured files under
ADSP-PHC-122024-investigator/. These are a separate NACC-linked harmonized
source in ID space A, used for modality and feature questions; they are not
needed for every NACC or ADNI question. The configured groups are:

| Group | Configured data files | Purpose |
| --- | --- | --- |
| Cognition | NACC_ADSP_PHC_Cognition_2024.csv | Harmonized cognition. |
| Fluid biomarker | NACC_ADSP_PHC_Biomarker_2024.csv | Fluid biomarker availability. |
| MRI | NACC_ADSP_PHC_T1_Freesurfer_2024.csv, NACC_ADSP_PHC_T1_MUSE_2024.xlsx, NACC_ADSP_PHC_FLAIR_2024.csv, NACC_ADSP_PHC_DTI_2024.csv | Structural, FLAIR, and diffusion imaging features. |
| PET | NACC_ADSP_PHC_Amyloid_Simple_2024.csv, NACC_ADSP_PHC_Amyloid_Detailed_2024.csv, NACC_ADSP_PHC_Tau_Simple_2024.csv, NACC_ADSP_PHC_Tau_Detailed_2024.csv | Amyloid and tau PET features. |
| Neuropathology | NACC_ADSP_PHC_Neuropath_2024.csv | Neuropathology features. |
| Vascular risk | NACC_ADSP_PHC_VascularRisk_2024.csv | Vascular risk features. |

Use the release and permissions applicable to your research. The
[ADSP-PHC project page](https://adsp.niagads.org/funded-programs/phenotype-harmonization/)
describes its harmonized domains; this code's paths specifically target the
December 2024 R3-style NACC-linked files. Newer releases require checking
filenames, fields, and linkage before changing the configuration.

## Before running

Keep licensed data outside Git. The repository ignores the NACC/, ADNI/, and
ADSP-PHC-122024-investigator/ directories. Check your own agreement before
sharing derived outputs. The project's ID spaces stay separate: NACC and the
separate ADSP-PHC R3 source use NACCID (A); ADNI and its ADSP_PHC_* tables use
RID/PTID (B). There is no verified crosswalk in this project.

