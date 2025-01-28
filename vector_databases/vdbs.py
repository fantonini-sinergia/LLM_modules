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
import math
import pandas as pd
import datasets
from vector_databases import file_processing

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
            as_excel,
            vdbs_path,
            **kwargs,
            ):
        self.vdbs = []
        self.get_embeddings_for_vdb = get_embeddings_for_vdb
        self.as_excel = as_excel
        print("as_excel", as_excel, " ", self.as_excel)
        if self.as_excel:
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
            if "add_words" not in kwargs:
                raise ValueError("The argument 'add_words' is required when 'as_axcel' is False.")
            if "add_words_nr_word_thr" not in kwargs:
                raise ValueError("The argument 'add_words_nr_word_thr' is required when 'as_axcel' is False.")
            if "chars_per_word" not in kwargs:
                raise ValueError("The argument 'chars_per_word' is required when 'as_axcel' is False.")
            add_words = kwargs["add_words"]
            add_words_nr_word_thr = kwargs["add_words_nr_word_thr"]
            self.chars_per_word = kwargs["chars_per_word"]
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
            # Calculate how many words a bunch is long
            self.words_per_bunch = [len(vdb["content"][0])/self.chars_per_word for vdb in self.vdbs]
            # Calculate how many bunches to add to each one retrieved
            self.add_bunches = []
            for i, _ in enumerate(self.words_per_bunch):
                if self.words_per_bunch[i] < add_words_nr_word_thr:
                    add_bunches = int(add_words/self.words_per_bunch[i])
                    self.words_per_bunch[i] += add_bunches*self.words_per_bunch[i]
                    self.add_bunches.append(add_bunches)
                else:
                    self.add_bunches.append(False)
        print("databases turned into vector databases")

    @classmethod
    def from_files_list(
        cls,
        files,
        get_embeddings_for_vdb,
        as_excel,
        **kwargs,
    ):
        if as_excel:
            if "vect_columns" not in kwargs:
                raise ValueError("The argument 'vect_columns' is required when 'as_axcel' is True.")
            
            excel_dbs = [pd.read_excel(f["path"]).astype(str) for f in files]
            dbs = [datasets.Dataset.from_pandas(db) for db in excel_dbs]

        else:
            if "vdbs_params" not in kwargs:
                raise ValueError("The argument 'vdbs_params' is required when 'as_axcel' is False.")
            if "chars_per_word" not in kwargs:
                raise ValueError("The argument 'chars_per_word' is required when 'as_axcel' is False.")
            if "add_words" not in kwargs:
                raise ValueError("The argument 'add_words' is required when 'as_axcel' is False.")
            if "add_words_nr_word_thr" not in kwargs:
                raise ValueError("The argument 'add_words_nr_word_thr' is required when 'as_axcel' is False.")
            
            vdbs_params = kwargs["vdbs_params"]
            chars_per_word = kwargs["chars_per_word"]

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
                chars_per_bunch = vdb_params["words_per_bunch"]*chars_per_word
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
            as_excel,
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
        as_excel = parameters["as_excel"]
        vect_columns = parameters["vect_columns"]
        chars_per_word = parameters["chars_per_word"]
        return cls(
            dbs, 
            get_embeddings_for_vdb,
            as_excel, 
            vdbs_path,
            vect_columns = vect_columns,
            chars_per_word = chars_per_word,
            **kwargs        
            )

    def get_rag_samples(
            self,
            text,
            get_embeddings_for_question, 
            context_word_len = None,
            ):

        """
        args:
        - 
        """
        # embed the question
        embededded_question = get_embeddings_for_question(text)
        print("Question embedded")

        if self.as_excel:
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

            # calculate the number of bunches to retrieve from each vdb
            nr_retrieved = context_word_len/sum(self.words_per_bunch)

            # calculate how many words to take for the last sample
            ratio_last_bunch = nr_retrieved - int(nr_retrieved)
            words_in_last_bunch = [int(w*ratio_last_bunch) for w in self.words_per_bunch]

            samples_per_vdb = []
            for i, vdb in enumerate(self.vdbs):
                
                # retrieve the samples for every vdb
                int_nr_retrieved = math.ceil(nr_retrieved)
                _, nearest_exs = vdb.get_nearest_examples(
                    "embeddings", 
                    embededded_question, 
                    k = int_nr_retrieved
                    )
                print(f'Retrieved {int_nr_retrieved} samples for vdb nr {i + 1}')
                
                # extend the samples of every vdb
                nearest_exs = extend_bunches(
                    vdb, 
                    nearest_exs, 
                    self.add_bunches[i]
                    )
                
                # cut the last sample of every vdb
                chars_in_last_bunch = words_in_last_bunch[i]*self.chars_per_word
                nearest_exs["content"][-1] = nearest_exs["content"][-1][:chars_in_last_bunch]
                print(f"The last sample has been cut to {words_in_last_bunch[i]} words")

                samples_per_vdb.append(nearest_exs)

        return samples_per_vdb