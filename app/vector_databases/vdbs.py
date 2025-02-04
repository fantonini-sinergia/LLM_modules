"""
Class vdbs is a list of vector databases for retrieval augmented generation

- Structure: List(HuggingFace_dataset(
                "bunch_count": list(int) --- A counter that is incremented for each consecutive bunch of the dataset
                "split_count": list(int) --- A counter that is incremented for each consecutive split operation of the dataset
                "file_name": list(str) --- Name of the file the bunch was created from
                "file_path": list(str) --- Url of the file the bunch was created from
                "file_extension": list(str) --- Extension of the file the bunch was created from
                "content": list(str) --- content od the bunch
                "page": list(int) --- Page of the file the bunch was created from
    ))

- Functions for the initialization:
    -- from_files_list
    -- from_dir

- Functions for content retrieval:
    -- get_rag_samples

"""

import os
import json
import requests
import pandas as pd
import datasets
from app.vector_databases import file_processing

def extend_bunches(
        vdb, 
        nearest_exs, 
        add_bunches
        ):

    if add_bunches:
        for j, sample_bunch_count in enumerate(nearest_exs["bunch_count"]):

            # reduce nr bunches to be added until it's not out of range 
            if sample_bunch_count + add_bunches > vdb["bunch_count"][-1]:
                temp_add_bunches = vdb["bunch_count"][-1] - sample_bunch_count
            else:
                temp_add_bunches = add_bunches

            # reduce nr bunches to be added until it remains inside one split only
            while vdb["split_count"][sample_bunch_count+temp_add_bunches] != vdb["split_count"][sample_bunch_count]:
                temp_add_bunches -= 1

            # append the bunches
            nearest_exs["content"][j] = "".join(
                [vdb["content"][sample_bunch_count+k] for k in range(temp_add_bunches+1)]
                    )
        print('Samples extended')
        
    return nearest_exs

class Vdbs():
    def __init__(
            self, 
            dbs,
            get_embeddings_for_vdb,
            search,
            vdbs_path,
            **kwargs,
            ):
        self.vdbs = []
        self.get_embeddings_for_vdb = get_embeddings_for_vdb
        self.search = search
        print("search", search, " ", self.search)
        if self.search:
            if "vect_columns" not in kwargs:
                raise ValueError("The argument 'vect_columns' is required when 'as_axcel' is True.")
            vect_columns = kwargs["vect_columns"]
            if len(vect_columns) == 0:
                vect_columns = self.vdbs[0].column_names
            if vdbs_path == None:
                # New vdbs as excels
                for db in dbs:
                    vdb = db.map(
                            lambda batch: {f"{col}_embed": [
                                    get_embeddings_for_vdb(val) for val in batch[col]
                                    ] for col in vect_columns},
                            batched=True
                        )
                    for col in vect_columns:
                        vdb = vdb.add_faiss_index(column=f"{col}_embed")
                    print('New columns: ', vdb.list_indexes())
                    self.vdbs.append(vdb)
            else:
                # Loaded vdbs as excels
                for i, db in enumerate(dbs):
                    vdb = db
                    for col in vect_columns:
                        vdb.load_faiss_index(
                            f"{col}_embed", 
                            os.path.join(vdbs_path, f'{i}_{col}_embed.faiss')
                            )
                    self.vdbs.append(vdb)
        else:
            if "add_chars" not in kwargs:
                raise ValueError("The argument 'add_chars' is required when 'as_axcel' is False.")
            if "add_chars_nr_char_thr" not in kwargs:
                raise ValueError("The argument 'add_chars_nr_char_thr' is required when 'as_axcel' is False.")
            add_chars = kwargs["add_chars"]
            add_chars_nr_char_thr = kwargs["add_chars_nr_char_thr"]
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
            # Calculate how many bunches to add to each one retrieved
            self.add_bunches = []
            for i, _ in enumerate(self.chars_per_bunch):
                if self.chars_per_bunch[i] < add_chars_nr_char_thr:
                    add_bunches = int(add_chars/self.chars_per_bunch[i])
                    self.chars_per_bunch[i] += add_bunches*self.chars_per_bunch[i]
                    self.add_bunches.append(add_bunches)
                else:
                    self.add_bunches.append(False)
        print("databases turned into vector databases")

    @classmethod
    def from_files_list(
        cls,
        files,
        get_embeddings_for_vdb,
        search,
        **kwargs,
    ):
        if search:
            if "vect_columns" not in kwargs:
                raise ValueError("The argument 'vect_columns' is required when 'as_axcel' is True.")
            
            excel_dbs = [pd.read_excel(f["path"]).astype(str) for f in files]
            dbs = [datasets.Dataset.from_pandas(db) for db in excel_dbs]

        else:
            if "vdbs_params" not in kwargs:
                raise ValueError("The argument 'vdbs_params' is required when 'as_axcel' is False.")
            if "add_chars" not in kwargs:
                raise ValueError("The argument 'add_chars' is required when 'as_axcel' is False.")
            if "add_chars_nr_char_thr" not in kwargs:
                raise ValueError("The argument 'add_chars_nr_char_thr' is required when 'as_axcel' is False.")
            
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
            search,
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
        search = parameters["search"]
        vect_columns = parameters["vect_columns"]
        return cls(
            dbs, 
            get_embeddings_for_vdb,
            search, 
            vdbs_path,
            vect_columns = vect_columns,
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
        response = requests.get(api_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data from API. Status code: {response.status_code}")
        
        data = response.json()
        
        if search:
            if "vect_columns" not in kwargs:
                raise ValueError("The argument 'vect_columns' is required when 'as_axcel' is True.")
            
            excel_dbs = [pd.DataFrame(data).astype(str)]
            dbs = [datasets.Dataset.from_pandas(db) for db in excel_dbs]

        else:
            if "vdbs_params" not in kwargs:
                raise ValueError("The argument 'vdbs_params' is required when 'as_axcel' is False.")
            if "add_chars" not in kwargs:
                raise ValueError("The argument 'add_chars' is required when 'as_axcel' is False.")
            if "add_chars_nr_char_thr" not in kwargs:
                raise ValueError("The argument 'add_chars_nr_char_thr' is required when 'as_axcel' is False.")
            
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
                    samples.drop(columns='scores').to_dict(orient='list')
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
                
                # extend the samples of every vdb
                nearest_exs = extend_bunches(
                    vdb, 
                    nearest_exs, 
                    self.add_bunches[i]
                    )

                samples_per_vdb.append(nearest_exs)

        return samples_per_vdb
    
    def stack(self, other):
        if (self.get_embeddings_for_vdb == other.get_embeddings_for_vdb
            and self.add_bunches_nr_char_thr == other.add_bunches_nr_char_thr
            and not self.search 
            and not other.search):
            self.vdbs.extend(other.vdbs)
            self.chars_per_bunch.extend(other.chars_per_bunch)
            self.add_bunches.extend(other.add_bunches)
            print("VDBs merged successfully.")
        else:
            raise ValueError("The VDBs cannot be merged. Ensure that 'get_embeddings_for_vdb'\
                             is the same for both objects and 'search' is False for both objects.")