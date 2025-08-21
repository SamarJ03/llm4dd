import os 
import polars as pl
import numpy as np
import pubchempy as pcp
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from dotenv import set_key
import requests
import json
from utils import Log

class DataPrep:
    def __init__(self):
        self.logger = Log.setLog(name="dataPrep")
        self.base_path = os.getenv(
            "BASE_PATH",
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        )
        self.data_path = os.path.join(self.base_path, 'data')
        os.makedirs(self.data_path, exist_ok=True)

        feat_path = os.path.join(self.data_path, 'features')
        os.makedirs(feat_path, exist_ok=True)
        os.makedirs(os.path.join(feat_path, 'meta'), exist_ok=True)
        
        # Fix scaffold directory naming consistency
        scaf_path = os.path.join(self.data_path, 'scaffold')
        os.makedirs(scaf_path, exist_ok=True)
        os.makedirs(os.path.join(scaf_path, 'ecfp4'), exist_ok=True)
        os.makedirs(os.path.join(scaf_path, 'maccs'), exist_ok=True)
        os.makedirs(os.path.join(scaf_path, 'rdkit'), exist_ok=True)

        # Store paths for later use
        self.featPath = feat_path
        self.featMetaPath = os.path.join(feat_path, 'meta')
        self.scafPath = scaf_path

        # Define featureColumns and calc here
        self.featureColumns = {
            'rdkit': [desc[0] for desc in Descriptors.descList],
            'maccs': [f'MACCS_{i}' for i in range(167)],
            'ecfp4': [f'ECFP4_{i}' for i in range(2048)]
        }
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.featureColumns['rdkit'])

        # Initialize RDKit standardization components
        self.uncharger = rdMolStandardize.Uncharger()
        self.te = rdMolStandardize.TautomerEnumerator()
        
        # Set random state for reproducibility
        self.rs = int(os.getenv("RANDOM_STATE", "15"))

        libDF, features = self.standardize()
        for key, val in features.items():
            self.curateFeatures(df=pl.DataFrame(val), type=str(key))

    @logger.catch
    def ConfigureRawData(self, sourcePath:str=None) -> pl.DataFrame:
        source_path = sourcePath if sourcePath else os.getenv("SOURCE_PATH")
        if source_path is None: raise ValueError(f'"SOURCE_PATH" not included in arguments or found in .env')
        if not os.path.exists(source_path): raise FileNotFoundError(f'source_path is not a valid file path: {source_path}')

        _,ext = os.path.splitext(source_path); ext = ext.lower()
        if ext=='.csv': df = pl.read_csv(source_path, has_header=True)
        elif ext in ['.xlsx', '.xls']: df = pl.read_excel(source_path, has_header=True)
        elif ext=='.json': df = pl.read_json(source_path)
        elif ext=='.parquet': df = pl.read_parquet(source_path)
        else: raise NotImplementedError(f'Source file path is of an unsupported type: [{ext}]')

        if isinstance(df, pl.DataFrame):
            try:
                req = pl.DataFrame(df).select(['compound_name', 'SMILES', 'score'])
                opt = pl.DataFrame(df.select([optCol for optCol in ['pathway', 'target', 'info'] if optCol in df.columns]))
                rawData = req if not opt else pl.concat([req, opt], how='horizontal')
                rawData.write_csv(os.path.join(self.data_path, 'raw_lib.csv'), include_header=True)
                set_key(os.getenv("ENV_PATH"), "RAWDATA_PATH", os.path.join(self.data_path, 'raw_lib.csv'))
                return True
            except Exception as e: raise Exception(f'Error in ConfigureRawData(): {e}')

    @logger.catch
    def getRawData(self) -> pl.DataFrame:
        log = Log.setLog(name='DataPrep.getRawData()')
        path = os.getenv("RAWDATA_PATH")
        if not os.path.exists(path) or path is None: path = os.path.join(self.data_path, 'raw_lib.csv')
        if not os.path.exists(path) or path is None:
            log.error(f'Raw data file path not found: {path}')
            return None
        return pl.read_csv(path, has_header=True)

    @logger.catch
    def standardize(self, rawData:pl.DataFrame=None):
        if rawData is None:
            rawData = self.getRawData()

        @logger.catch
        def addFeatures(self, smi:str, nameKey:str) -> dict:
            log = self.logger
            try: mol = Chem.MolFromSmiles(smi)
            except Exception as e: log.debug(f'!-{nameKey}: feature extraction: could not configure Chem.Mol')
            if not mol:
                log.debug(f'!-{nameKey}: feature extraction: could not configure Chem.Mol')
                return None

            rdkitSer, maccsSer, ecfp4Ser = None, None, None
            try: rdkitSer = pl.Series(
                values=list(self.calc.CalcDescriptors(mol)),
                index=self.featureColumns['rdkit'],
                dtype=float
            )
            except Exception as e: log.debug(f'!-{nameKey}: feature extraction [rdkit]: {e}')
            try: maccsSer = pl.Series(
                values=list(MACCSkeys.GenMACCSKeys(mol)),
                index=self.featureColumns['maccs'],
                dtype=int
            )
            except Exception as e: log.debug(f'!-{nameKey}: feature extraction [maccs]: {e}')
            try: ecfp4Ser = pl.Series(
                values=list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)),
                index=self.featureColumns['ecfp4'],
                dtype=int
            )
            except Exception as e: log.debug(f'!-{nameKey}: feature extraction [ecfp4]: {e}')
            return {'rdkit': rdkitSer, 'maccs': maccsSer, 'ecfp4': ecfp4Ser}

        @logger.catch
        def normalize(self, nameKey, smi) -> tuple[bool, str]:
            log = self.logger
            log.debug(f'{nameKey}: Normalizing ')
            newsmi = None
            try: mol = Chem.MolFromSmiles(smi)
            except Exception as e:
                log.debug(f'!-{nameKey}: failed normalization (smi -> rdkit.Chem.Mol)')
                return False, str(e)
            try: cleanedMol = rdMolStandardize.Cleanup(mol)
            except Exception as e:
                log.debug(f'!-{nameKey}: failed normalization (H+ removal, metal atom disconnection, mol normalization+reionization)')
                return False, str(e)
            try: parentMol = rdMolStandardize.FragmentParent(cleanedMol)
            except Exception as e:
                log.debug(f'!-{nameKey}: failed normalization (retrieving parent fragment)')
                return False, str(e)
            try: unchargedMol = self.uncharger.uncharge(parentMol)
            except Exception as e:
                log.debug(f'!-{nameKey}: failed normalization (mol neutralization)')
                return False, str(e)
            try: tautMol = self.te.Canonicalize(unchargedMol)
            except Exception as e:
                log.debug(f'!-{nameKey}: failed normalization (tautomer canonicalization)')
                return False, str(e)
            try: newsmi = Chem.MolToSmiles(tautMol)
            except Exception as e:
                log.debug(f'!-{nameKey}: failed normalization (rdkit.Chem.Mol -> smi)')
                return False, str(e)
            return True, str(newsmi)

        log = self.logger
        parsed = {
            'nan': {
                'compound_name': [],
                'SMILES': []
            },
            'non-standardized': {},
            'duplicate': {},
            'invalid_score': {}
        }
        if rawData is not None: df = pl.DataFrame(rawData)
        else: df = pl.read_csv(os.path.join(self.data_path, 'raw_lib.csv'))
        if not {'compound_name', 'SMILES', 'score'}.issubset(df.columns): raise ValueError("Missing required columns")
        else: df = df.cast({'compound_name': str, 'SMILES': str, 'score': float})
        lib = pl.DataFrame(schema=df.columns)
        log.info(f'standardizing raw compound library: {df.shape}')

        rdkit, ecfp4, maccs = [], [], []

        index = 0
        for row in df.iter_rows():
            index += 1
            name = str(row['compound_name']).strip()
            smi = str(row['SMILES']).strip()
            pScore = row['score']
            pathway = row['pathway'] if 'pathway' in df.columns else None
            target = row['target'] if 'target' in df.columns else None
            info = row['info'] if 'info' in df.columns else None

            # NaN and outlier removal:
            if name is None or name.lower() in ['', 'nan', 'none', 'na']: name = '(default)'
            if smi is None or smi.lower() in ['', 'nan', 'none', 'na']: # remove nan SMILES value
                parsed['nan']['SMILES'].append(f'[{index}]_{name}')
                log.debug(f'!-[{index}]_{name}: NaN removal (nan smiles)\n')
                continue
            if pScore is None or str(pScore).lower() in ['', 'nan', 'none', 'na']: # remove nan score value
                parsed['nan']['score'].append(f'[{index}]_{name}')
                log.debug(f'!-[{index}]_{name}: NaN removal (nan score)\n')
                continue
            if pScore<=0: # remove invalid score value (0 <= score <= 1)
                parsed['invalid_score'][f'[{index}]_{name}'] = pScore
                log.debug(f'!-[{index}]_{name}: invalid score: {pScore}\n')
                continue
            nameKey = f'[{index}]_{name}'

            # SMILES normalization:
            norm = self.normalize(smi, nameKey)
            if norm[0]: newsmi = norm[1].strip()
            else:
                parsed['non-standardized'][nameKey] = norm[1]
                continue

            # search for name on pubchempy if not listed
            if name=='(default)':
                url = f"https://cactus.nci.nih.gov/chemical/structure/{smi}/iupac_name"
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    name = response.text
                except Exception as e: log.warning(f'Pubchem lookup failed: {e}')

            # remove duplicates
            dupRow = lib.filter(pl.col("SMILES")==newsmi)
            if not dupRow.is_empty or dupRow is not None:
                oldName = str(dupRow['compound_name'].item()); oldSmi = str(dupRow['SMILES'].item()); oldScore = float(dupRow['score'].item())
                if pScore < oldScore: # drop previous iteration
                    lib = lib.remove(compound_name=oldName, SMILES=oldSmi, score=oldScore)
                    parsed['duplicate'][f'{oldName}_{oldSmi}'] = oldScore
                    log.debug(f'!-[{oldName}_{oldSmi}]: duplicate removal (previous)')
                elif pScore > oldScore:
                    parsed['duplicate'][f'{name}_{smi}'] = pScore
                    logger.debug(f'!-{nameKey}: duplicate removal (current)')
                    continue
                elif name==oldName and smi==oldSmi and pScore==oldScore:
                    logger.debug(f'!-{nameKey}: exact duplicates; skipping current')
                    continue
            smi = newsmi

            # add binary labels (and optional int labels)
            binLabel = 1 if pScore < 1 else 0

            row = {
                'compound_name': name,
                'SMILES': smi,
                'score': pScore,
                'label': binLabel
            }
            if 'pathway' in df.columns: row['pathway'] = pathway
            if 'target' in df.columns: row['target'] = target
            if 'info' in df.columns: row['info'] = info
            lib = pl.concat([lib, pl.DataFrame(row)], how='horizontal')

            features = addFeatures(smi, nameKey)
            rdkit.append(pl.concat([row, pl.Series(features['rdkit'])], how='horizontal'))
            maccs.append(pl.concat([row, pl.Series(features['maccs'])], how='horizontal'))
            ecfp4.append(pl.concat([row, pl.Series(features['ecfp4'])], how='horizontal'))

        log.info(f'completed data standardization and SMILES normalization.')
        lib.write_csv(os.path.join(self.data_path, 'lib.csv'))
        with open(os.path.join(self.base_path, 'logs', 'filtered_compounds.json'), 'w') as f: json.dump(parsed, f, indent=4)

        log.info(f'completed feature extraction.')
        rdkitDF = pl.DataFrame(rdkit)
        rdkitDF.write_csv(os.path.join(self.featMetaPath, 'rdkit_meta.csv'))
        maccsDF = pl.DataFrame(maccs)
        maccsDF.write_csv(os.path.join(self.featMetaPath, 'maccs_meta.csv'))
        ecfp4DF = pl.DataFrame(ecfp4)
        ecfp4DF.write_csv(os.path.join(self.featMetaPath, 'ecfp4_meta.csv'))

        return lib, {
            'rdkitFeatures': rdkitDF,
            'maccsFeatures': maccsDF,
            'ecfp4Features': ecfp4DF
        }

    # drop low-var features and split into scaffold test/train/valid sets
    @logger.catch
    def curateFeatures(self, df: pl.DataFrame, type:str, vt:float=float(os.getenv("VARIANCE_THRESHOLD", "0.1"))):
        log = self.logger
        if type not in ['rdkit', 'maccs', 'ecfp4']: log.error("dataset type not supported: {type}")
        colToDrop = ['compound_name', 'SMILES', 'score', 'label', 'pathway', 'target', 'info']
        # infoDF = pl.DataFrame(df.filter([col for col in colToDrop in df.columns]))
        infoDF = pl.DataFrame(df.select([col for col in colToDrop if col in df.columns]))
        out = pl.DataFrame(df.drop([col for col in colToDrop if col in df.columns])).cast(int if type!='rdkit' else float)
        outnp = out.to_numpy()
        var = np.var(outnp, axis=0)
        lowVarCol = [out.columns[i] for i, variance in enumerate(var) if variance < vt]
        meta = pl.concat([infoDF, out], how='horizontal').drop(lowVarCol)
        meta.write_csv(os.path.join(self.featPath, f'{type}.csv'))

        currPath = os.path.join(self.scafPath, type); os.makedirs(currPath, exist_ok=True)

        def split(df: pl.DataFrame, name:str, path, rs:int=int(os.getenv("RANDOM_STATE", "15"))):
            drop_cols = ['compound_name', 'score', 'pathway', 'target', 'info']
            train, temp = train_test_split(df, test_size=0.2, random_state=rs, stratify=df['label'], shuffle=True)
            valid, test = train_test_split(temp, test_size=0.5, random_state=rs, stratify=temp['label'], shuffle=True)
            pl.DataFrame(train).drop(columns=[col for col in drop_cols if col in train.columns]).to_csv(f'{path}/{name}_train.csv', index=False)
            pl.DataFrame(test).drop(columns=[col for col in drop_cols if col in test.columns]).to_csv(f'{path}/{name}_test.csv', index=False)
            pl.DataFrame(valid).drop(columns=[col for col in drop_cols if col in valid.columns]).to_csv(f'{path}/{name}_valid.csv', index=False)

        def splitSubcats(name, path, sc:dict=None):
            for key, val in sc.items():
                if str(key)=='all': split(df=val, name=name, path=path)
                else:
                    subPath = os.path.join(path, key); os.makedirs(subPath, exist_ok=True)
                    split(df=val, name=key, path=subPath)

        if type=='rdkit':
            descs = pl.Series(list(out.columns))
            estateCols = [col for col in descs.to_list() if "EState" in str(col)]
            fgcCols = [col for col in estateCols if col.startswith('fr_')]
            mtCols = [col for col in fgcCols if str(col).lower() in ['balabanj', 'bertzct', 'hallkieralpha', 'ipc', 'avgipc']]
            fbCols = [col for col in mtCols if any(str(col).startswith(prefix) for prefix in ['Fp', 'BCUT2D'])]
            saCols = [col for col in fbCols if any(x in str(col).lower() for x in ['peoe', 'smr', 'slogp']) or str(col).lower()=='labuteasa']
            sdcCols = [
                col for col in saCols if (
                    str(col).lower().startswith('n') and str(col).lower() not in [
                        'numvalenceelectrons', 'numradicalelectrons'
                    ]
                ) or str(col).lower() in ['fractioncsp3', 'ringcount']
            ]
            pcCol = [col for col in descs.to_list() if col not in estateCols+fgcCols+mtCols+fbCols+saCols+sdcCols]
            subcategories = {
                'all': pl.DataFrame(meta),
                'EState': pl.concat([infoDF, pl.DataFrame(out).filter(estateCols)], how='horizontal'),
                'functionalGroupCounts': pl.concat([infoDF, pl.DataFrame(out).filter(fgcCols)], how='horizontal'),
                'molecularTopology': pl.concat([infoDF, pl.DataFrame(out).filter(mtCols)], how='horizontal'),
                'fingerprintBased': pl.concat([infoDF, pl.DataFrame(out).filter(fbCols)], how='horizontal'),
                'surfaceArea': pl.concat([infoDF, pl.DataFrame(out).filter(saCols)], how='horizontal'),
                'structural': pl.concat([infoDF, pl.DataFrame(out).filter(sdcCols)], how='horizontal'),
                'physiochemical': pl.concat([infoDF, pl.DataFrame(out).filter(pcCol)], how='horizontal'),
            }
            splitSubcats(name=type, path=currPath, sc=subcategories, rs=self.rs)
        else: split(df=meta, name=type, path=currPath, rs=self.rs)

if __name__ == "__main__":
    try:
        import yaml
        import os
        
        # Load config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get source path from config
        source_path = config['env']['source_path']
        
        # Initialize data preparation
        data_prep = DataPrep()
        
        # Configure raw data with source path from config
        if source_path and source_path.lower() != "local":
            data_prep.ConfigureRawData(sourcePath=source_path)
        
        # Process will run standardization and feature extraction automatically
        # as it's part of the initialization
        logger.info("Data preparation completed successfully")
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")

