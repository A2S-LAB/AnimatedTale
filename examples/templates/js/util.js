const get_skeleton = () => {
    var form = new FormData();
    form.append( "file", $("#fileInput")[0].files[0] );
    $.ajax({
        type:"POST",
        url:"/process_skeleton",
        data:form,
        processData : false,
        contentType : false,
        success:function(result){
            console.log(result);

            const image = $("#uploadedImage")
            const svg = $("#svg")
            const width = image.width()
            const height = image.height()
            svg.css("display", "block")
            svg.width(width + "px")
            svg.height(height + "px")

            draw_coordinate(result)
        }
    })
}

const draw_coordinate = (e) => {
    coor = e.coordinate
    shape = e.shape
    width_rate = $("#uploadedImage").width() / shape[0]
    height_rate = $("#uploadedImage").height() / shape[1]

    $.each(coor, function(idx, val){
        console.log(val)
        info = {}
        info.id = idx
        info.x = val[0] * width_rate
        info.y = val[1] * height_rate

        draw_circle(info)
    })
}

const draw_circle = (info) => {
    let tagString =
        `<circle
            id='${info.id}'
            cx='${info.x}'
            cy='${info.y}'
            fill='#ff0000'
            r='6'
        />`
    document.getElementById('svg').appendChild(parseSVG(tagString))
}

const parseSVG = (s) => {
    let div= document.createElementNS('http://www.w3.org/1999/xhtml', 'div')
    div.innerHTML= '<svg xmlns="http://www.w3.org/2000/svg">'+s+'</svg>'
    let frag= document.createDocumentFragment()
    while (div.firstChild.firstChild)
        frag.appendChild(div.firstChild.firstChild)
    return frag
}
