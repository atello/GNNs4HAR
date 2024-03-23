from torch_geometric.data import Data


def create_data_pairs(source: Data, target: Data):
    data = PairData(x_s=source.x, edge_index_s=source.edge_index, edge_weight_s=source.edge_weight, y_s=source.y,
                    x_t=target.x, edge_index_t=target.edge_index, edge_weight_t=target.edge_weight, y_t=target.y)
    return data


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)
