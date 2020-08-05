"""
   Copyright 2020 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import tensorflow as tf

from datanetAPI import DatanetAPI


POLICIES = {'WFQ':0, 'SP':1, 'DRR':2}
def detect_scenario(w1,tos,policy):
    
    WFQ_only = True
    Scenario = 1
    '''Filter Scenario 1 and 2 where all links have WFQ policy'''
    for i in policy:
        if i!=POLICIES['WFQ']:
            WFQ_only = False
            break
    '''Mark as Scenario 2 if weight-1 for any link is not 60 ''' 
    if(WFQ_only):
        for i in w1:
            if i!=60:
                Scenario=2
                

    else:
        count = [0,0,0]
        Scenario=3
        for i in tos:
            count[int(i)]+=1
        ''' If Percentage of ToS=0 is closer to 33.3% than 10% its Scenario 4'''
        if (count[0]/sum(count) > 0.2166):
            Scenario=4
    
    return Scenario
         
def binary_search(arr, x): 
    low = 0
    high = len(arr) - 1
    mid = 0
  
    while low <= high: 
  
        mid = (high + low) // 2
  
        # Check if x is present at mid 
        if arr[mid] < x: 
            low = mid + 1
  
        # If x is greater, ignore left half 
        elif arr[mid] > x: 
            high = mid - 1
  
        # If x is smaller, ignore right half 
        else: 
            return mid 
  
    # If we reach here, then the element was not present 
    return -1       
            
count = [0,0]     
def generator(data_dir, shuffle = False):
    global count
    """This function uses the provided API to read the data and returns
       and returns the different selected features.

    Args:
        data_dir (string): Path of the data directory.
        shuffle (string): If true, the data is shuffled before being processed.

    Returns:
        tuple: The first element contains a dictionary with the following keys:
            - bandwith
            - packets
            - link_capacity
            - links
            - paths
            - sequences
            - n_links, n_paths
            The second element contains the source-destination delay
    """
    final = np.genfromtxt("final1.txt")
    tool = DatanetAPI(data_dir, [], shuffle)
    it = iter(tool)
    n=0
    maxf = max(final)
    for sample in it:
        ###################
        #  EXTRACT PATHS  #
        ###################
        if (n> maxf):
            print("break")
            break
        if binary_search(final,n)==-1:
            n+=1
            continue
    
        n+=1
        
        routing = sample.get_routing_matrix()

        nodes = len(routing)
        # Remove diagonal from matrix
        paths = routing[~np.eye(routing.shape[0], dtype=bool)].reshape(routing.shape[0], -1)
        paths = paths.flatten()

        ###################
        #  EXTRACT LINKS  #
        ###################
        g = sample.get_topology_object()

        cap_mat = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
        q_policy = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=None)
        w_1 = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=0.0)
        w_2 = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=0.0)
        w_3 = np.full((g.number_of_nodes(), g.number_of_nodes()), fill_value=0.0)

        for node in range(g.number_of_nodes()):
            curr = g.nodes[node]
            for adj in g[node]:
                cap_mat[node, adj] = g[node][adj][0]['bandwidth']
                q_policy[node, adj] = POLICIES[curr['schedulingPolicy']]
                try:
                    weights = curr['schedulingWeights']
                except:
                    continue
                
                temp = weights.split(',')
                w_1[node,adj] =  float(temp[0])
                w_2[node,adj] =  float(temp[1])
                w_3[node,adj] =  float(temp[2])
                

        links = np.where(np.ravel(cap_mat) != None)[0].tolist()

        link_capacities = (np.ravel(cap_mat)[links]).tolist()

        q_policy = (np.ravel(q_policy)[links]).tolist()
        w_1 = (np.ravel(w_1)[links]).tolist()
        w_2 = (np.ravel(w_2)[links]).tolist()
        w_3 = (np.ravel(w_3)[links]).tolist()
        #print(links, link_capacities)

        ids = list(range(len(links)))
        links_id = dict(zip(links, ids))

        #print(len(ids), len(q_policy))

        #print(links_id)

        path_ids = []
        for path in paths:
            new_path = []
            for i in range(0, len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                new_path.append(links_id[src * nodes + dst])
            path_ids.append(new_path)

        ###################
        #   MAKE INDICES  #
        ###################
        link_indices = []
        path_indices = []
        sequ_indices = []
        segment = 0
        for p in path_ids:
            link_indices += p
            path_indices += len(p) * [segment]
            sequ_indices += list(range(len(p)))
            segment += 1

        #print(len(link_indices) , len(q_policy))

        traffic = sample.get_traffic_matrix()
        # Remove diagonal from matrix
        traffic = traffic[~np.eye(traffic.shape[0], dtype=bool)].reshape(traffic.shape[0], -1)

        # print(traffic.shape)

        result = sample.get_performance_matrix()
        # Remove diagonal from matrix
        result = result[~np.eye(result.shape[0], dtype=bool)].reshape(result.shape[0], -1)


        avg_bw = []
        pkts_gen = []
        delay = []
        type_of_service = []
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                flow = traffic[i, j]['AggInfo']
                type_of_service.append(traffic[i,j]['Flows'][0]['ToS'])
                avg_bw.append(flow['AvgBw'])
                pkts_gen.append(flow['PktsGen'])
                d = result[i, j]['AggInfo']['AvgDelay']
                delay.append(d)

        n_paths = len(path_ids)
        n_links = max(max(path_ids)) + 1
        Scenario = detect_scenario(w_1,type_of_service,q_policy)
        #print(n_links, len(w_1))
        if(n_paths==552):
            count[0]+=1
        else:
            count[1]+=1

        yield {"bandwith": avg_bw, "packets": pkts_gen,
               "link_capacity": link_capacities,
               "links": link_indices,
               "paths": path_indices, "sequences": sequ_indices,
               "n_links": n_links, "n_paths": n_paths, "ToS": type_of_service, 
               "Q_policy": q_policy, "w1": w_1, "w2": w_2, "w3": w_3 , "Scenario":Scenario}, delay


def transformation(x, y):
    """Apply a transformation over all the samples included in the dataset.

        Args:
            x (dict): predictor variable.
            y (array): target variable.

        Returns:
            x,y: The modified predictor/target variables.
        """
    return x, y


def input_fn(data_dir, transform=True, repeat=True, shuffle=False):
    """This function uses the generator function in order to create a Tensorflow dataset

        Args:
            data_dir (string): Path of the data directory.
            transform (bool): If true, the data is transformed using the transformation function.
            repeat (bool): If true, the data is repeated. This means that, when all the data has been read,
                            the generator starts again.
            shuffle (bool): If true, the data is shuffled before being processed.

        Returns:
            tf.data.Dataset: Containing a tuple where the first value are the predictor variables and
                             the second one is the target variable.
        """
    ds = tf.data.Dataset.from_generator(lambda: generator(data_dir=data_dir, shuffle=shuffle),
                                        ({"bandwith": tf.float32, "packets": tf.float32,
                                          "link_capacity": tf.float32, "links": tf.int64,
                                          "paths": tf.int64, "sequences": tf.int64,
                                          "n_links": tf.int64, "n_paths": tf.int64, "ToS": tf.float32, 
                                          "Q_policy": tf.float32,"w1": tf.float32,"w2": tf.float32,"w3": tf.float32},
                                        tf.float32),
                                        ({"bandwith": tf.TensorShape([None]), "packets": tf.TensorShape([None]),
                                          "link_capacity": tf.TensorShape([None]),
                                          "links": tf.TensorShape([None]),
                                          "paths": tf.TensorShape([None]),
                                          "sequences": tf.TensorShape([None]),
                                          "n_links": tf.TensorShape([]),
                                          "n_paths": tf.TensorShape([]), "ToS": tf.TensorShape([None]),
                                           "Q_policy": tf.TensorShape([None]), "w1": tf.TensorShape([None]),
                                           "w2": tf.TensorShape([None]),"w3": tf.TensorShape([None])},
                                         tf.TensorShape([None])))
    if transform:
        ds = ds.map(lambda x, y: transformation(x, y))

    if repeat:
        ds = ds.repeat()

    return ds
