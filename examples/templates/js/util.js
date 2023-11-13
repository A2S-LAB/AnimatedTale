let mouse_state = 0     // 0 : mouse up , 1 : mouse down
let mouse_coor = {x:0, y:0}
let select_circle = -1

let joints = []
let contours = []
let width_rate = 1
let height_rate = 1

let mode = "joint" //segment

const get_skeleton = () => {
    var form = new FormData();
    return_val = false
    form.append( "file", $("#fileInput")[0].files[0]);
    $.ajax({
        type:"POST",
        url:"/process_skeleton",
        data:form,
        processData : false,
        contentType : false,
        async : false,
        success:function(result){
            const image = $("#uploadedImage")
            const svg = $("#svg")
            const width = image.width()
            const height = image.height()
            svg.css("display", "block")
            svg.width(width + "px")
            svg.height(height + "px")

            draw_joint(result)
        }
    })
}

const draw_joint = (e) => {
    $('circle').remove()

    shape = e.shape
    joints = e.joints
    contours = e.contours
    width_rate = $("#uploadedImage").width() / shape[0]
    height_rate = $("#uploadedImage").height() / shape[1]

    $.each(joints, function(idx, val){
        if(idx == 0) return true
        info = {}
        info.id = idx
        info.x = val[0] * width_rate
        info.y = val[1] * height_rate
        info.fill = "#ff0000"

        draw_circle(info)
        document.getElementById(`${idx}`).addEventListener('mousedown', function(){
            select_circle = idx
        })
    })
}

const draw_contours = (e) => {
    info = {}
    info.id = "polygon"

    let cal_contours = []
    $.each(contours, function(idx, val){
        cor = val[0]
        cal_contours.push([[cor[0] * width_rate, cor[1] * height_rate]])
    })
    info.points = cal_contours

    drawPolygon(info)
}

const drawPolygon = (info) => {
    let tagString =
        `<polygon
            id='${info.id}'
            points='${info.points}'
            style='
                stroke:#ff1105;
                fill:#ff0000;
                fill-opacity:0.6';
        />`
    document.getElementById('svg').appendChild(parseSVG(tagString))
}

const draw_circle = (info) => {
    let tagString =
        `<circle
            id='${info.id}'
            cx='${info.x}'
            cy='${info.y}'
            fill='${info.fill}'
            r='6'
        />`
    document.getElementById('svg').appendChild(parseSVG(tagString))
}

const move_circle = (info) => {
    select_circle = info
}

const parseSVG = (s) => {
    let div= document.createElementNS('http://www.w3.org/1999/xhtml', 'div')
    div.innerHTML= '<svg xmlns="http://www.w3.org/2000/svg">'+s+'</svg>'
    let frag= document.createDocumentFragment()
    while (div.firstChild.firstChild)
        frag.appendChild(div.firstChild.firstChild)
    return frag
}
