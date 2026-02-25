"""
ESM2 特征提取遗留模块 / ESM2 Feature Extraction Legacy Module
=============================================================
本文件保留旧版类，供验证模型正确性或历史对比使用。
不在主工作流中调用，但根据原作者要求（v0.020前不允许删除）予以保留。

This file preserves legacy classes for model verification or historical comparison.
Not used in the main workflow. Kept per original author's requirement (do not delete before v0.020).

包含 / Contains:
    - Esm2LastHiddenFeatureExtractor_legacy
    - Esm2LayerHiddenFeatureExtractor_legacy

迁移自 / Migrated from: ESM2_fr.py (lines 1232–1702)
"""

import pandas as pd
from packaging import version
import warnings
import numpy as np
from itertools import islice

try:
    import torch
    if version.parse(torch.__version__) < version.parse('1.12.1'):
        warnings.warn("Your torch version is older than 1.12.1 and may not operate correctly.")
except ImportError:
    warnings.warn("Torch not found. Some functions will not be available.")

try:
    import tqdm
    from tqdm import tqdm
except ImportError:
    warnings.warn("tqdm is not installed. Some features may not work as expected.")


class Esm2LastHiddenFeatureExtractor_legacy:
    def __init__(self, tokenizer, model, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False,device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")

    def get_last_hidden_states(self, outputs):
        last_hidden_state = outputs.hidden_states[-1]
        return last_hidden_state

    def get_last_cls_token(self, last_hidden_state):
        return last_hidden_state[:, 0, :]

    def get_last_eos_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, eos_position, :]

    def get_last_mean_token(self, last_hidden_state, eos_position):
        return last_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, last_hidden_state, eos_position, num_segments=10):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, num_segments)
        segment_means = []

        start = 1
        for i in range(num_segments):
            end = start + segment_size + (1 if i < remainder else 0)

            if end > start:
                segment_mean = last_hidden_state[:, start:end, :].mean(dim=1)
            else:
                segment_mean = torch.zeros(last_hidden_state[:, start:start+1, :].shape, device=last_hidden_state.device)

            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means

    ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
    def get_last_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}
        columns = None
        all_results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    last_hidden_state = self.get_last_hidden_states(outputs)
                    last_cls_token = self.get_last_cls_token(last_hidden_state[j:j+1]) if self.compute_cls else None
                    last_eos_token = self.get_last_eos_token(last_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    last_mean_token = self.get_last_mean_token(last_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(last_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    features = []
                    if last_cls_token is not None:
                        cls_features = last_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if last_eos_token is not None:
                        eos_features = last_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if last_mean_token is not None:
                        mean_features = last_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        X_outcome = pd.concat(all_results, axis=0)
        print(f'Features dimensions: {features_length}')
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result

    ##计算磷酸化表征
    def get_last_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence',
                                                         phosphorylation_positions='phosphorylation_positions', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        self.model.to(self.device)

        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    last_hidden_state = self.get_last_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)
                        position_feature = last_hidden_state[:, position, :]
                        features = position_feature.squeeze().tolist()
                        X_outcome.loc[idx] = features

                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result

    def get_amino_acid_representation(self, sequence, amino_acid, position):
        self.model.to(self.device)
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = self.get_last_hidden_states(outputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_position = position if amino_acid == tokens[position] else -1

        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")

        amino_acid_features = last_hidden_state[:, token_position, :].squeeze().tolist()
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T
        return amino_acid_features_df


class Esm2LayerHiddenFeatureExtractor_legacy:
    def __init__(self, tokenizer, model, layer_indicat, compute_cls=True, compute_eos=True, compute_mean=True, compute_segments=False, device_choose = 'auto'):
        self.tokenizer = tokenizer
        self.model = model
        self.layer_indicat = layer_indicat
        self.compute_cls = compute_cls
        self.compute_eos = compute_eos
        self.compute_mean = compute_mean
        self.compute_segments = compute_segments

        if device_choose == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_choose == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise TypeError("CUDA is not available. Please check your GPU settings.")
        elif device_choose == 'cpu':
            self.device = torch.device("cpu")

    def get_layer_hidden_states(self, outputs):
        layer_hidden_state = outputs.hidden_states[self.layer_indicat]
        return layer_hidden_state

    def get_layer_cls_token(self, layer_hidden_state):
        return layer_hidden_state[:, 0, :]

    def get_layer_eos_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, eos_position, :]

    def get_layer_mean_token(self, layer_hidden_state, eos_position):
        return layer_hidden_state[:, 1:eos_position, :].mean(dim=1)

    def get_segment_mean_tokens(self, layer_hidden_state, eos_position, num_segments=10):
        seq_len = eos_position - 1
        segment_size, remainder = divmod(seq_len, num_segments)
        segment_means = []

        start = 1
        for i in range(num_segments):
            end = start + segment_size + (1 if i < remainder else 0)

            if end > start:
                segment_mean = layer_hidden_state[:, start:end, :].mean(dim=1)
            else:
                segment_mean = torch.zeros(layer_hidden_state[:, start:start+1, :].shape, device=layer_hidden_state.device)

            segment_means.append(segment_mean.squeeze().tolist())
            start = end

        return segment_means

    ##计算cls, eos, 氨基酸平均表征, 每1/10段氨基酸平均表征
    def get_layer_hidden_features_combine(self, X_input, sequence_name='sequence', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        self.model.to(self.device)
        sequence = X_input[sequence_name].tolist()

        features_length = {}
        columns = None
        all_results = []
        with torch.no_grad():
            for i in tqdm(range(0, len(sequence), batch_size), desc='batches for inference'):
                batch_sequences = sequence[i:i+batch_size]
                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j in range(len(batch_sequences)):
                    idx = i + j
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    eos_position = tokens.index(self.tokenizer.eos_token) if self.tokenizer.eos_token in tokens else len(batch_sequences[j])
                    layer_hidden_state = self.get_layer_hidden_states(outputs)
                    layer_cls_token = self.get_layer_cls_token(layer_hidden_state[j:j+1]) if self.compute_cls else None
                    layer_eos_token = self.get_layer_eos_token(layer_hidden_state[j:j+1], eos_position) if self.compute_eos else None
                    layer_mean_token = self.get_layer_mean_token(layer_hidden_state[j:j+1], eos_position) if self.compute_mean else None
                    segment_means = self.get_segment_mean_tokens(layer_hidden_state[j:j+1], eos_position) if self.compute_segments else None

                    features = []
                    if layer_cls_token is not None:
                        cls_features = layer_cls_token.squeeze().tolist()
                        if 'cls' not in features_length:
                            features_length['cls'] = len(cls_features)
                        features.extend(cls_features)

                    if layer_eos_token is not None:
                        eos_features = layer_eos_token.squeeze().tolist()
                        if 'eos' not in features_length:
                            features_length['eos'] = len(eos_features)
                        features.extend(eos_features)

                    if layer_mean_token is not None:
                        mean_features = layer_mean_token.squeeze().tolist()
                        if 'mean' not in features_length:
                            features_length['mean'] = len(mean_features)
                        features.extend(mean_features)

                    if segment_means is not None:
                        for seg, segment_mean in enumerate(segment_means):
                            features.extend(segment_mean)
                            if f'segment{seg}_mean' not in features_length:
                                features_length[f'segment{seg}_mean'] = len(segment_mean)

                    if columns is None:
                        columns = []
                        for feature_type, length in features_length.items():
                            for k in range(length):
                                columns.append(f"ESM2_{feature_type}{k}")

                    result = pd.DataFrame([features], columns=columns, index=[idx])
                    all_results.append(result)

                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        X_outcome = pd.concat(all_results, axis=0)
        print(f'Features dimensions: {features_length}')
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result

    ##计算磷酸化表征
    def get_layer_hidden_phosphorylation_position_feature(self, X_input, sequence_name='sequence', phosphorylation_positions='phosphorylation_positions', batch_size=32):
        X_input = X_input.reset_index(drop=True)
        self.model.to(self.device)

        grouped_X_input = X_input.groupby(sequence_name)
        sequence_to_indices = grouped_X_input.groups

        num_features = self.model.config.hidden_size
        columns = [f"ESM2_phospho_pos{k}" for k in range(num_features)]
        X_outcome = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for i in tqdm(range(0, len(grouped_X_input), batch_size), desc='batches for inference'):
                batch_sequences = list(islice(sequence_to_indices.keys(), i, i + batch_size))
                batch_grouped_sequences = {seq: X_input.loc[sequence_to_indices[seq]] for seq in batch_sequences}

                inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model(**inputs)

                for j, sequence in enumerate(batch_sequences):
                    sequence_indices = batch_grouped_sequences[sequence].index
                    sequence_positions = batch_grouped_sequences[sequence][phosphorylation_positions].tolist()
                    layer_hidden_state = self.get_layer_hidden_states(outputs)[j:j+1]

                    for idx, position in zip(sequence_indices, sequence_positions):
                        position = int(position)
                        position_feature = layer_hidden_state[:, position, :]
                        features = position_feature.squeeze().tolist()
                        X_outcome.loc[idx] = features

                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        print(f"The dimension of the final phosphorylation features is: {X_outcome.shape[1]}")
        combined_result = pd.concat([X_input, X_outcome], axis=1)
        return combined_result

    def get_amino_acid_representation(self, sequence, amino_acid, position):
        self.model.to(self.device)
        if sequence[position - 1] != amino_acid:
            raise ValueError(f"The amino acid at position {position} is not {amino_acid}.")

        inputs = self.tokenizer([sequence], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        layer_hidden_state = self.get_layer_hidden_states(outputs)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_position = position if amino_acid == tokens[position] else -1

        if token_position == -1:
            raise ValueError(f"The token for amino acid {amino_acid} could not be found in the tokenized sequence.")

        amino_acid_features = layer_hidden_state[:, token_position, :].squeeze().tolist()
        feature_names = [f"ESM2_{k}" for k in range(len(amino_acid_features))]
        amino_acid_features_df = pd.DataFrame(amino_acid_features, index=feature_names, columns=[amino_acid]).T
        return amino_acid_features_df
