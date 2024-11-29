import os
import math
import numpy as np
import datasets
from file_reading import read_file

def split_in_bunches(
        file: dict, 
        chars_per_bunch: int, 
        resplits: int, 
        all_bunches_counter: int,
        all_splits_counter: int
        ):

    bunches_content = []
    bunches_start_page = []
    bunches_counter = []
    splits_counter = []
    text = file["text"]

    # split per page for excels
    if file["file_extension"].upper() in ['XLSX', 'XLS']:
        pages_start_char = file["pages_start_char"]
        for page, char in enumerate(pages_start_char, start=1):
            bunches_content.append(text[:char])
            bunches_start_page.append(page)

            # update counters
            bunches_counter.append(all_bunches_counter)
            splits_counter.append(all_splits_counter)
            all_bunches_counter += 1
        all_splits_counter += 1

    # split per chars interval for every other file type
    else:
        pages_start_char = np.array(file["pages_start_char"]).flatten()
        num_split_operations = resplits + 1

        # if text too short, round the bunch length
        min_bunch_over_text_ratio = num_split_operations/(2*num_split_operations-1)
        if chars_per_bunch > len(text)*min_bunch_over_text_ratio:
            chars_per_bunch = round(len(text)*min_bunch_over_text_ratio)
        
        # for every split operations
        for _ in range(num_split_operations):

            # create bunches
            bunches_per_split_op = [text[i*chars_per_bunch:i*chars_per_bunch+chars_per_bunch] for i in range(round(len(text)/chars_per_bunch))]
            nr_new_bunches = len(bunches_per_split_op)

            # calculate start pages
            pages_per_split_op = []
            for i in range(len(bunches_per_split_op)):
                distances = pages_start_char - i*chars_per_bunch
                pages_per_split_op.append(1 + distances[distances < 0].size)
            bunches_content += bunches_per_split_op
            bunches_start_page += pages_per_split_op
            
            # prepare for next iteration
            new_file_start = round(chars_per_bunch*1/num_split_operations)
            text = text[new_file_start:]
            pages_start_char = pages_start_char - new_file_start
            pages_start_char = pages_start_char[pages_start_char > 0]

            # update counters
            bunches_counter += [i for i in range(all_bunches_counter, all_bunches_counter+nr_new_bunches)]
            splits_counter += [all_splits_counter]*nr_new_bunches
            all_bunches_counter += nr_new_bunches
            all_splits_counter += 1
        
    return {
        "bunches_content": bunches_content,
        "bunches_counter": bunches_counter,
        "splits_counter": splits_counter,
        "bunches_start_page": bunches_start_page
    }

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
        vdbs_params
    ):
        
        """
        original files format
        List of dicts with args:
            -str 'name'
            -str 'path'
        """

        # read every file
        for i, file in enumerate(files):
            files[i] = {**file, **read_file(file)}
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
                bunches = split_in_bunches(
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