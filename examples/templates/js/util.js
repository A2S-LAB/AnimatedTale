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
        }
    })
}