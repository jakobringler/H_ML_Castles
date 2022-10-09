samples = len(upstream_items)

# read attributes

for upstream_item in upstream_items:
    sampleIndex = samples - 1
    if upstream_item.index > sampleIndex:
        pass
    else:
        work_item = upstream_item
        
        # extract attributes
        
        i = int(work_item.index)
        dir = str(work_item.attribValue("directory"))
        file = str(work_item.attribValue("filename"))
        
        dir = dir.replace("/hip/../", "/")
        
        path = os.path.join(dir, file)
       
        print(path)