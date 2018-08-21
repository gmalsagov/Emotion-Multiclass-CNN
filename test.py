has_wildcard = False
wildcard_index = 0
product_non_wildcard_dims = 1

shape_data = [-1, 40, 300, 1]

for i in range(len(shape_data)):
    if shape_data[i] == -1:
        if not has_wildcard:
            has_wildcard = True
            wildcard_index = i
    else:
        product_non_wildcard_dims *= shape_data[i]

print product_non_wildcard_dims
print has_wildcard
print wildcard_index