let mouse_state = 0     // 0 : mouse up , 1 : mouse down
let mouse_coor = {x:0, y:0}
let select_circle = -1

let joints = []
let joint_label = []
let contours = []
let gif_name = ""
let width_rate = 1
let height_rate = 1

let mode = "joint" //segment

const get_skeleton = () => {
    var form = new FormData();
    form.append("file", $("#fileInput")[0].files[0]);
    $.ajax({
        type:"POST",
        url:"/process_skeleton",
        data:form,
        processData : false,
        contentType : false,
        success:function(result){
            const image = $("#uploadedImage")
            const svg = $("#svg")
            const width = image.width()
            const height = image.height()
            svg.css("display", "block")
            svg.width(width + "px")
            svg.height(height + "px")

            draw_joint(result)
            contours = result.contours

            button_count()
        }
    })
}

const button_count = () => {
    let buttonVal = ($("#submitButton").attr("value") * 1) + 1
    $("#submitButton").attr("value", buttonVal )
}

const make_gif = () => {
    var form = new FormData();

    form.append("file", $("#fileInput")[0].files[0]);
    form.append("gif_name", JSON.stringify({"gif_name":gif_name}))
    form.append("contour", JSON.stringify({"contour":contours}))
    form.append("joint", JSON.stringify({"joint":joints}))

    $.ajax({
        type:"POST",
        url:"/make_gif",
        data:form,
        processData : false,
        contentType : false,
        success:function(result){
            console.log(result)
            if(result == 'done') window.location.href = 'http://172.30.1.84:8888/exhibit'
        }
    })
}

const predict_sam = () => {
    var form = new FormData();
    const req_joints = { "joints" : joints }
    const req_labels = { "labels" : joint_label }
    form.append("file", $("#fileInput")[0].files[0])
    form.append("joints", JSON.stringify(req_joints))
    form.append("labels", JSON.stringify(req_labels))

    $.ajax({
        type:"POST",
        url:"/process_sam",
        data:form,
        dataType:"json",
        processData : false,
        contentType : false,
        success:function(result){
            contours = result.contours
            draw_contours()
        }
    })
}

const draw_joint = (e) => {
    $('circle').remove()

    shape = e.shape
    joints = e.joints
    joint_text = e.joint_text
    width_rate = $("#uploadedImage").width() / shape[0]
    height_rate = $("#uploadedImage").height() / shape[1]

    $.each(joints, function(idx, val){
        if(idx == 0) return true
        info = {}
        info.id = idx
        info.x = val[0] * width_rate
        info.y = val[1] * height_rate
        info.fill = "#ff0000"
        info.text = joint_text[idx]
        draw_circle(info)
        draw_text(info)

        const circleElement = document.getElementById(`c-${idx}`);

        if (circleElement) {
            circleElement.addEventListener('mousedown', function() {
                select_circle = idx;
            });
    
            circleElement.addEventListener('touchstart', function(event) {
                select_circle = idx;
                mouse_state = 1;
                event.preventDefault();
            });
        } else {
            console.log(`Element with id c-${idx} not found`);
        }
    })
}



const draw_contours = (e) => {
    $("polygon").remove()
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

const draw_text = (e) => {
    let tagString =
        `<text
            id='${info.id}'
            x='${info.x}'
            y='${info.y}'

        >${info.text}</text>`
        document.getElementById('svg').appendChild(parseSVG(tagString))
}

const drawPolygon = (info) => {
    let tagString =
        `<polygon
            id='${info.id}'
            points='${info.points}'
            style='
                stroke:#ff1105;
                fill:#FFE4C4;
                fill-opacity:0.5';
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

const parseSVG = (s) => {
    let div= document.createElementNS('http://www.w3.org/1999/xhtml', 'div')
    div.innerHTML= '<svg xmlns="http://www.w3.org/2000/svg">'+s+'</svg>'
    let frag= document.createDocumentFragment()
    while (div.firstChild.firstChild)
        frag.appendChild(div.firstChild.firstChild)
    return frag
}
