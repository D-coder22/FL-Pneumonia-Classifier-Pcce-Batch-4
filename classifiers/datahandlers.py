import os


def get_datahandler_config(dh_name, folder_data, party_id, is_agg):
    if dh_name == 'xray':
        data = {
            'name': 'XrayKerasDataHandler',
            'path': 'ibmfl.util.data_handlers.xray_keras_data_handler',
            'info': {
                'npz_file': os.path.join('data', 'data_party' + str(party_id) + '.npz')
            }
        }
        if is_agg:
            data['info'] = {
                'npz_file': os.path.join("classifiers", "data", "xray.npz")
            }


    return data
