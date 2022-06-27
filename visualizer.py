def visualize(all_iter_origs, all_iter_tags, origs, preds, file_path):
    with open(file_path, 'w') as fp:
        for all_iter_orig, all_iter_tag, orig_sent, pred_tokens \
            in zip(all_iter_origs, all_iter_tags, origs, preds):
            n_iter = 0
            # print('Original:', ' '.join(orig_tokens))
            # print('Prediction:', ' '.join(pred_tokens))
            fp.write('Original:   ' + orig_sent + '\n')
            fp.write('Prediction: ' + ' '.join(pred_tokens) + '\n')
            for orig_toekns, tags in zip(all_iter_orig, all_iter_tag):
                if ''.join(tags) == '':
                    continue
                n_iter += 1
                # print(f'----- Iteration {n_iter} -----')
                fp.write(f'----- Iteration {n_iter} -----\n')
                tokens_print = []
                tags_print = []
                for token, tag in zip(orig_toekns, tags):
                    max_len = max(len(token), len(tag))
                    tokens_print.append(token + ' '*(max_len - len(token)))
                    tags_print.append(tag + ' '*(max_len - len(tag)))
                # print(' '.join(tokens_print))
                # print(' '.join(tags_print))
                fp.write(' '.join(tokens_print) + '\n')
                fp.write(' '.join(tags_print) + '\n')
            # print('\n==========\n')
            fp.write('\n==========\n')
    return