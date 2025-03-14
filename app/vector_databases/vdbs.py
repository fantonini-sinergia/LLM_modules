import os
import json
import requests
import pandas as pd
import datasets
from app.vector_databases import file_processing

class Vdbs():
    def __init__(
            self, 
            dbs,
            get_embeddings_for_vdb,
            vdbs_path,
            **kwargs,
            ):
        self.vdbs = []
        self.get_embeddings_for_vdb = get_embeddings_for_vdb
        if vdbs_path == None:
            # New vdbs as raw files
            for db in dbs:
                vdb = db.map(
                        lambda x: {"embeddings": get_embeddings_for_vdb(x["content"])}
                ).add_faiss_index(column="embeddings")
                self.vdbs.append(vdb)
        else:
            # Loaded vdbs as raw files
            for i, db in enumerate(dbs):
                vdb = db
                vdb.load_faiss_index(
                    'embeddings', 
                    os.path.join(vdbs_path, f'{i}.faiss')
                    )
            self.vdbs.append(vdb)
            # Calculate how many chars a bunch is long
            self.chars_per_bunch = [len(vdb["content"][0]) for vdb in self.vdbs]
        print("databases turned into vector databases")

    @classmethod
    def from_files_list(
        cls,
        files,
        get_embeddings_for_vdb,
        **kwargs,
    ):

        if "vdbs_params" not in kwargs:
            raise ValueError("from_files_list: The argument 'vdbs_params' is required.")
        
        vdbs_params = kwargs["vdbs_params"]

        # read every file
        """
        original files format
        List of dicts with args:
            -str 'name'
            -str 'path'
        """
        for i, file in enumerate(files):
            files[i] = {**file, **file_processing.read_file(file)}
        print(f"Readed the files")
        """
        new files format
        List of dicts with args:
            -str 'name'
            -str 'path'
            -str 'text'
            -str 'file_extension'
            -list 'pages_start_char'
        """

        # remove files without text
        files = [file for file in files if file["text"]]
        print("Removed files without text")

        # for every parameters set, for every file
        dbs = []
        for vdb_params in vdbs_params:
            chars_per_bunch = int(vdb_params["chars_per_bunch"])
            content_field = []
            page_field = []
            file_name_field = []
            file_path_field = []
            file_extension_field = []
            bunch_count_field = []
            split_count_field = []
            all_bunches_counter = 0
            all_splits_counter = 0
            for file in files:

                # generate the bunches
                bunches = file_processing.split_in_bunches(
                    file = file, 
                    chars_per_bunch = chars_per_bunch,
                    resplits = vdb_params["resplits"],
                    all_bunches_counter = all_bunches_counter,
                    all_splits_counter = all_splits_counter
                )
                num_bunches = len(bunches["bunches_content"])
                print(f'generated {num_bunches} text bunches for parameters set {vdb_params}')

                # load bunches content
                content_field += bunches["bunches_content"]

                # load bunches counts
                bunch_count_field += bunches["bunches_counter"]
                all_bunches_counter = bunch_count_field[-1] + 1

                # load splits counts
                split_count_field += bunches["splits_counter"]
                all_splits_counter = split_count_field[-1] + 1

                # load bunches start page
                page_field += bunches["bunches_start_page"]

                # load bunches file
                file_name_field += [file["name"]]*num_bunches

                # load bunches file path
                file_path_field += [file["path"]]*num_bunches

                # load bunches file extension
                file_extension_field += [file["file_extension"]]*num_bunches

                # append to the db
                dbs.append(datasets.Dataset.from_dict({
                    "bunch_count": bunch_count_field,
                    "split_count": split_count_field,
                    "file_name": file_name_field,
                    "file_path": file_path_field,
                    "file_extension": file_extension_field,
                    "content": content_field,
                    "page": page_field,
                    }))

        return cls(
            dbs,
            get_embeddings_for_vdb,
            None,
            **kwargs
            )
            
    
    @classmethod
    def from_dir(
        cls,
        vdbs_path,
        get_embeddings_for_vdb,
        **kwargs,
        ):
        dbs = []
        for dir in os.listdir(vdbs_path):
            if ".hf" in dir:
                dbs.append(datasets.Dataset.load_from_disk(os.path.join(vdbs_path, dir)))
                print(f"\n\nthat's the dir\n{dir}\n\n")
            else:
                print(f"\n\nthis is not a dir\n{dir}\n\n")
        with open(os.path.join(vdbs_path,"parameters.json"), "r") as file:
            parameters = json.load(file)
        return cls(
            dbs, 
            get_embeddings_for_vdb,
            vdbs_path,
            **kwargs        
            )
    
    @classmethod
    def from_api(
        cls,
        api_url,
        get_embeddings_for_vdb,
        search,
        **kwargs,
        ):
        response = requests.get(api_url, params={"crafterSite": "ideale"})
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data from API. Status code: {response.status_code}")
        
        data = response.json()
        
        if search:
            if "vect_columns" not in kwargs:
                raise ValueError("The argument 'vect_columns' is required when 'search' is True.")
            
            # excel_dbs = [pd.DataFrame(data).astype(str)]

            """
            PROVVISORIO PER CARLONI
            """
            excel_dbs = [pd.DataFrame([{
                **{'nome': item['name_s']},
                **{'contenuto': item['contenuto_t']},
                **{'obiettivo': item['obiettivo_t']},
                **{'partner': item['partner_o']['item'][0]['component']['name_s']},
                **{'costo': item['costo_s']},
                # **{'extra_cost': item['extra_costo_t']},
                **{'durata': item['durata_s']},
                **{'modalità': item['modalita_t']},
                **{'target': item['rivolto_t']},
                **{'url': item['rootId']},
                **{'max partecipanti': item['maxpartecipanti_s']},
                **{'macrocategoria': item['subcategory_o']['item'][0]['component']['category_o']['item'][0]['component']['macro_category_o']['item'][0]['component']['name_s']},
                **{'categoria': item['subcategory_o']['item'][0]['component']['category_o']['item'][0]['component']['name_s']},
                **{'sottocategoria': item['subcategory_o']['item'][0]['component']['name_s']},
                **{'tags': ', '.join([tag['value_smv'] for tag in item['subcategory_o']['item'][0]['component']['tags_o']['item']])},
                } for item in data['items']])]

            dbs = [datasets.Dataset.from_pandas(db) for db in excel_dbs]

        else:
            """
            lascia questa parte per dopo Carloni
            """
            if "vdbs_params" not in kwargs:
                raise ValueError("The argument 'vdbs_params' is required when 'search' is False.")
            
            vdbs_params = kwargs["vdbs_params"]

            # Process the data
            files = [{"name": f"api_data_{i}", "path": api_url, "text": item["text"], "file_extension": "json", "pages_start_char": [0]} for i, item in enumerate(data)]
            print(f"Processed the API data")

            # for every parameters set, for every file
            dbs = []
            for vdb_params in vdbs_params:
                chars_per_bunch = int(vdb_params["chars_per_bunch"])
                content_field = []
                page_field = []
                file_name_field = []
                file_path_field = []
                file_extension_field = []
                bunch_count_field = []
                split_count_field = []
                all_bunches_counter = 0
                all_splits_counter = 0
                for file in files:

                    # generate the bunches
                    bunches = file_processing.split_in_bunches(
                        file = file, 
                        chars_per_bunch = chars_per_bunch,
                        resplits = vdb_params["resplits"],
                        all_bunches_counter = all_bunches_counter,
                        all_splits_counter = all_splits_counter
                    )
                    num_bunches = len(bunches["bunches_content"])
                    print(f'generated {num_bunches} text bunches for parameters set {vdb_params}')

                    # load bunches content
                    content_field += bunches["bunches_content"]

                    # load bunches counts
                    bunch_count_field += bunches["bunches_counter"]
                    all_bunches_counter = bunch_count_field[-1] + 1

                    # load splits counts
                    split_count_field += bunches["splits_counter"]
                    all_splits_counter = split_count_field[-1] + 1

                    # load bunches start page
                    page_field += bunches["bunches_start_page"]

                    # load bunches file
                    file_name_field += [file["name"]]*num_bunches

                    # load bunches file path
                    file_path_field += [file["path"]]*num_bunches

                    # load bunches file extension
                    file_extension_field += [file["file_extension"]]*num_bunches

                # append to the db
                dbs.append(datasets.Dataset.from_dict({
                    "bunch_count": bunch_count_field,
                    "split_count": split_count_field,
                    "file_name": file_name_field,
                    "file_path": file_path_field,
                    "file_extension": file_extension_field,
                    "content": content_field,
                    "page": page_field,
                    }))

        return cls(
            dbs,
            get_embeddings_for_vdb,
            search,
            None,
            **kwargs
            )

    def get_rag_samples(
            self,
            text,
            get_embeddings_for_question, 
            nr_bunches = 1,
            ):

        """
        """
        # embed the question
        embededded_question = get_embeddings_for_question(text)
        print("Question embedded")

        if self.search:
            # retrieve the samples, search case
            """
            In case of search, there is a single vdb only (a list is used to mantain the same
            structure of search = False). However, we have more than one retrieval, one per vect_column.
            There is no difference in the length of the samples when retrieving with a different vect_column,
            so at the end of the loop, the samples are concatenated and sorted by the scores.
            """
            samples_per_vdb = []
            samples = pd.DataFrame()
            for i, vdb in enumerate(self.vdbs):
                for vect_col in vdb.list_indexes():
                    sc, sa = vdb.get_nearest_examples(
                        vect_col, 
                        embededded_question, 
                        k = 3
                        )
                    sa = pd.DataFrame.from_dict(sa)
                    sa['scores'] = sc
                    sa['from'] = [vect_col]*len(sc)
                    samples = pd.concat([samples, sa], ignore_index=True)
                samples = samples.sort_values(by='scores')[:3]
                samples_per_vdb.append(
                    # samples.drop(columns='scores').to_dict(orient='list') # output as columns
                    samples.drop(columns=['scores', vect_col, "from"]).to_dict(orient='records') #output as rows
                    )

        else:
            # retrieve the samples, search=False case
            """
            In case of search=False, we have a list of vdbs. A retrieval for each vdb is made, and,
            each time, the n samples with high scores are taken. The number of samples from each vdb is
            decided by the parameter nr_bunches. The samples are then extended.
            """
            samples_per_vdb = []
            for i, vdb in enumerate(self.vdbs):
                
                # retrieve the samples for every vdb
                _, nearest_exs = vdb.get_nearest_examples(
                    "embeddings", 
                    embededded_question, 
                    k = nr_bunches,
                    )
                print(f'Retrieved {nr_bunches} samples for vdb nr {i + 1}')

                samples_per_vdb.append(nearest_exs)

        return samples_per_vdb
    
    def stack(self, other):
        if (self.get_embeddings_for_vdb == other.get_embeddings_for_vdb
            and not self.search 
            and not other.search):
            self.vdbs.extend(other.vdbs)
            self.chars_per_bunch.extend(other.chars_per_bunch)
            print("VDBs merged successfully.")
        else:
            raise ValueError("The VDBs cannot be merged. Ensure that 'get_embeddings_for_vdb'\
                             is the same for both objects and 'search' is False for both objects.")