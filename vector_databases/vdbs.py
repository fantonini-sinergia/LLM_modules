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
import math
import pandas as pd
import datasets
from . import file_processing

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

class Vdbs:
    def __init__(
            self, 
            vdbs_list,
            get_embeddings_for_vdb,
            chars_per_word,
            vdbs_path = None,
            ):
        self.vdbs = vdbs_list
        self.chars_per_word = chars_per_word
        
        if vdbs_path == None:
            for i in range(len(self.vdbs)):
                self.vdbs[i] = self.vdbs[i].map(
                        lambda x: {"embeddings": get_embeddings_for_vdb(x["content"])}
                ).add_faiss_index(column="embeddings")
        else:
            for i, vdb in enumerate(self.vdbs):
                vdb.load_faiss_index('embeddings', f'{vdbs_path}\\faiss_{i}.faiss')

        print("databases turned into vector databases")

    @classmethod
    def from_files_list(
        cls,
        files, 
        get_embeddings_for_vdb,
        chars_per_word,
        vdbs_params,
        as_excel = False,
    ):
        if as_excel:
            excel_dbs = [pd.read_excel(f).astype(str) for f in files]
            vdbs = [datasets.Dataset.from_pandas(db) for db in excel_dbs]

        else:
            """
            original files format
            List of dicts with args:
                -str 'name'
                -str 'path'
            """

            # read every file
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
            vdbs = []
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
                vdbs.append(datasets.Dataset.from_dict({
                    "bunch_count": bunch_count_field,
                    "split_count": split_count_field,
                    "file_name": file_name_field,
                    "file_path": file_path_field,
                    "file_extension": file_extension_field,
                    "content": content_field,
                    "page": page_field,
                    }))

        return cls(vdbs, get_embeddings_for_vdb, chars_per_word)
    
    @classmethod
    def from_dir(
        cls,
        vdbs_path,
        get_embeddings_for_vdb,
        chars_per_word
        ):
        
        vdbs = []
        for dir in os.listdir(vdbs_path):
            if "database" in dir:
                vdbs.append(datasets.Dataset.load_from_disk(os.path.join(vdbs_path, dir)))
        print(f'Loaded {len(vdbs)} Databases. Number of bunches per db:')
        for perm_vdb in vdbs:
            print(f'- {len(perm_vdb["page"])}')
        print("-"*20, "\n\n")
        return cls(vdbs, get_embeddings_for_vdb, chars_per_word, vdbs_path = vdbs_path)
    
    @classmethod
    def from_xlsx(
        cls,
        xlsx_path,
        get_embeddings_for_vdb,
        chars_per_word,
    ):

        df = pd.read_excel(xlsx_path)

        # append to the db
        vdbs = [datasets.Dataset.from_pandas(df)]

        return cls(vdbs, get_embeddings_for_vdb, chars_per_word)


    def get_rag_samples(
            self,
            text,
            get_embeddings_for_question, 
            context_word_len,
            add_words,
            add_words_nr_word_thr
            ):

        """
        args:
        - 
        """

        # calculate the number of bunches to be retrieved (decimal, same for all vdbs)
        words_per_bunch_per_vdb = [len(vdb["content"][0])/self.chars_per_word for vdb in self.vdbs]
        add_bunches_per_vdb = []
        for i, _ in enumerate(words_per_bunch_per_vdb):
            if words_per_bunch_per_vdb[i] < add_words_nr_word_thr:
                add_bunches = int(add_words/words_per_bunch_per_vdb[i])
                words_per_bunch_per_vdb[i] += add_bunches*words_per_bunch_per_vdb[i]
                add_bunches_per_vdb.append(add_bunches)
            else:
                add_bunches_per_vdb.append(False)
        nr_retrieved = context_word_len/sum(words_per_bunch_per_vdb)

        # calculate how many words to take for the last sample of every db
        ratio_last_bunch_per_vdb = nr_retrieved - int(nr_retrieved)
        words_in_last_bunch_per_vdb = [int(w*ratio_last_bunch_per_vdb) for w in words_per_bunch_per_vdb]

        
        # embed the question
        embededded_question = get_embeddings_for_question(text)
        print("Question embedded")

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
                add_bunches_per_vdb[i]
                )
            
            # cut the last sample of every vdb
            chars_in_last_bunch = words_in_last_bunch_per_vdb[i]*self.chars_per_word
            nearest_exs["content"][-1] = nearest_exs["content"][-1][:chars_in_last_bunch]
            print(f"The last sample has been cut to {words_in_last_bunch_per_vdb[i]} words")

            samples_per_vdb.append(nearest_exs)

        return samples_per_vdb