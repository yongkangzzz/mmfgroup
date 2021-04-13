function loading() {
  localStorage.setItem(
      "methods",
      document.getElementById("selectMethod").value
  );
  document.getElementById("loader").style.display = "block";
  console.log(document.getElementById("loader").style.display);
  document.getElementById("bglogo").style.display = "none";
  document.getElementById("txtVisl").style.display = "none";
  document.getElementById("infArea").innerHTML = "Predicting...Please wait...";
}

function stashTextInput() {
  console.log("txt changed")
  localStorage.setItem(
      "txtInput",
      document.getElementById("textInput").value
  );
}

function showOptionForm(that) {
  if (that.value === "internalModel") {
    document.getElementById("internalModel").style.display = "block";
    document.getElementById("selfModel").style.display = "none";
    document.getElementById("noModel").style.display = "none";
  } else if (that.value === "selfModel") {
    document.getElementById("internalModel").style.display = "none";
    document.getElementById("selfModel").style.display = "block";
    document.getElementById("noModel").style.display = "none";
  } else if (that.value === "noModel") {
    document.getElementById("internalModel").style.display = "none";
    document.getElementById("selfModel").style.display = "none";
    document.getElementById("noModel").style.display = "block";
  } else {
    document.getElementById("internalModel").style.display = "none";
    document.getElementById("selfModel").style.display = "none";
    document.getElementById("noModel").style.display = "none";
  }
}

// function storeMethod(that) {
//   localStorage.setItem('methods', that.value);
// }

window.addEventListener("DOMContentLoaded", (event) => {
  console.log("DOM fully loaded and parsed");
  var selectedVal = localStorage.getItem("methods");
  if (selectedVal) {
    document.getElementById("selectMethod").value = selectedVal;
    // localStorage.removeItem("methods");
  }
  var txtInput = localStorage.getItem("txtInput");
  if (txtInput) {
    document.getElementById("textInput").value = txtInput;
    localStorage.removeItem("txtInput");
  }
});

let tut = document.querySelector("#btnTutorial");
tut.onclick = function () {
  setTimeout(() => {
    const driver = new Driver();
    const stepList = [
      {
        element: "#btnUploadMdl",
        popover: {
          title: "Step 1",
          description:
              "Click here to select a model or upload your own checkpoints",
        },
      },
      {
        element: "#btnUploadImg",
        popover: {
          title: "Step 2",
          description: "Click here to select an image and upload",
        },
      },
      {
        element: "#imgBox1",
        popover: {
          title: "Then",
          description:
              "After uploading, your selected image will be displayed here",
        },
      },
      {
        element: "#btnInpaint",
        popover: {
          title: "Next: An Optional Choice, Text Removal",
          description:
              "If you want to remove texts that appears on the uploaded image, click this button",
        },
      },
      {
        element: "#textInput",
        popover: {
          title: "Step 3",
          description:
              "After uploading the image, add some texts to the image here",
        },
      },
      {
        element: "#selectMethod",
        popover: {
          title: "Step 4",
          description: "Select a method for explaining the model prediction",
        },
      },
      {
        element: "#selectExpDir",
        popover: {
          title: "Step 5",
          description: "If you want to see what factors in image/texts support model's prediction, select 'Encourage'. If you want to see what factors are against model's prediction, select 'Discourage'",
        },
      },
      {
        element: "#btnPredict",
        popover: {
          title: "Step 6",
          description: "Press this Button to run the explainable algorithm",
        },
      },
      {
        element: "#imgBox2",
        popover: {
          title: "Finally",
          description:
              "It may take about 2 minutes to run the interpretable algorithm. The result image will be shown here.",
        },
      },
      {
        element: "#textarea",
        popover: {
          title: "Finally",
          description: "Meanwhile the explanation for texts will be shown here.",
        },
      },
    ];
    driver.defineSteps(stepList);
    driver.start();
  }, 50);
};
