hparams = {
    'lr': 5e-5,
    'batch_size': 64,
    'student': {
        'pretrained': 'bert-base-uncased',
        'pretrained_dim': 768,
        'num_aspect': 9,
    },
    'description': 'bag_and_cases baseline',
    'save_dir': './LeverageJustAFewKeywords/ckpt/bags_and_cases',
    'aspect_init_file': './LeverageJustAFewKeywords/data/bags_and_cases.30.txt',
    'train_file': './LeverageJustAFewKeywords/data/bags_and_cases_train.json',
    'test_file': './LeverageJustAFewKeywords/data/bags_and_cases_test.json',
    'general_asp': 4,
    'maxlen': 40
}

# in local machine/gitlab repo, LeverageJustAFewKeywords is not the root folder