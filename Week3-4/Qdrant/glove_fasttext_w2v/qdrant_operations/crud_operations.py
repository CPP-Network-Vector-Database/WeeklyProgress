def update_vector(collection_name, point_id, new_vector):
    client.update_point(
        collection_name=collection_name,
        point_id=point_id,
        vector=new_vector
    )

def delete_point(collection_name, point_id):
    client.delete(collection_name, point_id=point_id)
