window.addEventListener(`resize`, function() {
    const width = canvas.offsetWidth
    const height = canvas.offsetHeight
    const padding = 60
    if (imgElement.width > imgElement.height){
        imgElement.style.width = (width - padding)+ "px";
    }else{
        imgElement.style.height = (height - padding)+ "px";
    }
});

document.oncontextmenu = function(){return false}

window.addEventListener('mousemove', function(e){
    if(mouse_state == 1 && select_circle != -1){
        X = e.layerX
        Y = e.layerY

        $(`#${select_circle}`).css("cx", X)
        $(`#${select_circle}`).css("cy", Y)
    }
})

window.addEventListener('mousedown', (e) => {
    if(mode == "joint"){
        if(e.target.classList[0] == undefined){
            mouse_state = 1
        }
        if(e.target.classList[0] == "canvas"){
            mouse_state = 0
        }
    }else if(mode == "segment"){
        if(e.which == 1){
            // left click
            info = {}
            info.x = e.layerX
            info.y = e.layerY
            info.fill = "#ff0000"
        }else if(e.which == 3){
            // right click
            info = {}
            info.x = e.layerX
            info.y = e.layerY
            info.fill = "#0000ff"
        }
        draw_circle(info)
    }
})

window.addEventListener('mouseup', function(e){
    mouse_state = 0
    select_circle = -1
})