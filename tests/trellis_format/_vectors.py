"""Golden Trellis v2 decode vectors encoded by ``piper_kernels.trellis``.

Each vector is one seeded Gaussian block encoded into little-endian bitstreams
with two trailing wrap guard bytes per block and checked against the portable
INT8 decode reference (``decode_reference(..., round_int8=True)``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchVector:
    """One encoded block: layout, FP8 E4M3 scale bits, stream and expected INT8 values."""

    block_size: int
    block_bits: int
    scale_code: int
    bitstream_hex: str
    values_hex: str


RESEARCH_VECTORS = (
    ResearchVector(
        256,
        768,
        100,
        (
            "dd781073e03f9d89e240f02e9035b004511020478e98ab7c9be9a067ca71e326025bfd3187b0b9ae"
            "566f380c70aac92c7307dc17983538c96902e0db37bd9a2593453bc29773b1f78e518d900dec3f5b"
            "bf7d47fe5953973cf15fe28b904930b4dd78"
        ),
        (
            "f7113206170ae1c8def80825f10652dd2abe1738b4c01f0107d9e619f8fa03f8ab14f5f31c1af2dd"
            "50142fe7e7ea28261d050d0616c7f3240fc147daf91b111313bcc40af3e70c0534e208df391ed226"
            "71e1be7fb61bb4052debf80f1810e7d1c3314812e718240a41c9d7f8f5c4fe1d25ef0de4f8c45bfb"
            "d01ceca3e1ece82bce430956b60dd7b7faba7fb4f440d0c80d41d7f21f38b6f1dbfcdc29e13ba732"
            "0c080c0e3f190d0f0acbdaf110f009cac2e00d030afb1b3d950e3dcd500c1fb7fe1ac4e359b89a9f"
            "e9e6f7dffb1a0f252015e013d8d3ce32de39efe0ddfb0c21995211043e1621d6cb1dd926f1183e2e"
            "7a48ee03dd0ecb2bf323dad1fcce063e"
        ),
    ),
    ResearchVector(
        256,
        888,
        100,
        (
            "8a2e0139d38b70ede84232a57fed96784aae3ce42c2038fbd6b46936aa4120f03f29344190f95d99"
            "fde9bbb1d5f544c47662d5ce8d2c34f79e9dbe5aac48f39fcd981821771bc0df8f2216a22002c0e4"
            "b676d5ae5c821de9edd710a20e9d8549bb0dadc85933d18de90d7835d574b28a2e"
        ),
        (
            "a7f1ff015050d6e80fc6cd73e6de30da3ee9f006da2e5acecf4f51f71510e316ed377f0f2965d916"
            "0b8104ace101bece399f34293fefc0fb0d0b2507f61fd9c446e1f313e773fe0ef1ba4de75ca31220"
            "290f33c622352c5b060e20fcb61529d713fb41d7f12f1ff9c8f8f5ec334c1c1b295145c20e09e9e8"
            "0f1fe0c4a501e32fe1d7f71a39f7cf973020f40af10707612310e80714fd0ffec31fe9650e01170b"
            "1d36ef0e47b0c9fe4f2fede215232c3ef0bd3bb2a7f72e20b30d092bf726a3c6d5bff90606d4fcd4"
            "44d900e6d0edececa70d161d340f091dcc3215ff953e07f112fbf8281d031b42eff04d1608f1bb0a"
            "01fe13ac09dcc3e1fbb1ba5edf062b24"
        ),
    ),
    ResearchVector(
        256,
        1024,
        98,
        (
            "1c24e1c7abb1fe1768a0e8d1882a102996c53b2dbc28994ab262b6c0ef6e7b26dbdf0a5ed1f873ea"
            "fdda94b0e4e269fb4231ad0cd17291ff092f9d386ad8b48076cbe5510c3325982c5f5d71d8744095"
            "b9df3d73d2d12bac0cbaaf1a08068a42b96c4107cd8b76b83572193b31a7f39c4dadac1c92647058"
            "558b5d920a4501b01c24"
        ),
        (
            "d0ff40ed08f81dcd08b05df618ddf4240ba10e0f4c49ed3b24fb47cf282b351576bebe0337e5df39"
            "221410ceff020af423c5f20f0fc8db29f814bde62105011910adfb201cec06fae6280d43f927ef3a"
            "c9fce7f1d535073b05392a0bd92f284f1bf600201ecfdcde2cd405d4fbe518f31511390d36eec811"
            "9e092711b6b851db1f16f10ff146d822fc42bdc8133b3816f0d8de05e8dfcdd5f016f4d548aaebfa"
            "4cd8edfb20dd18153afb0f18d2d60de40be4102b0aab18179e11a1f29c053aa123fa4f19da0919de"
            "1f14f31a492ae04e16e216d1f901321de50ed80121fe1b0eeb301ed0f8caee04d109c1401d0fff65"
            "34df4722140e230ee6024604ca54b0f7"
        ),
    ),
    ResearchVector(
        256,
        1144,
        100,
        (
            "63af2539f1ac579202da3684bf63cbdd223e18ffbeca926bebd099db556fb25c93624ccc56dcaf15"
            "7279bb27ecf735532b5cbfaa65cf4083799fca7b6178eb7a574a106146252131db6911a75066e27e"
            "c5e83f20b40dd0ff17205d05d33f83aac4faacb6cb7c9287198f1864e0b7b52fc55ee684fb52bbb0"
            "d73298744a8fc3c7aa298994f9b5da594a2d267c8e9fcc63af"
        ),
        (
            "23fad4d8ed0b0cf9002c142504b6b5662c46e92515f5202ff7fbc6d8c41742d2290c11ffcb18dd26"
            "eb10043111fb1ddde8b7d8e4ac21f0ca8efa1618fda9f0fb6123150eff77d39133f4a9eddc0fe3bb"
            "2430c410f5d1f2a5f509e32e0c19f7f93c0fdddd3de2af0fecf9c7b7f1bb18e72223fd06091842a7"
            "37bd02d2cde9273521fb06eef4e20eed0013e0fb150ee68dd91ce920ebced8b0eaee0011f0df0a32"
            "b106e8c31f4004ee1f48c3ece117e32aff16f6d707de6d140315cfcc032ee8dbe0e78fdd07243605"
            "2a7f37f8f128de9e1502e04de408083641fa4327133657faf8c7e013ddeff63fe41506dd0ca1ffd"
            "5add7fc4e0372a70408b8b231fe234ad8"
        ),
    ),
    ResearchVector(
        224,
        1000,
        98,
        (
            "ce7799b357b991c97502a1ea94971ace67fe3538df23c941012f217acfdcf24c7c4da196a42710ba"
            "b8dda3a9325995469265ad161c663f2ee3ef672077eef5632a21c7efcc4f821356fa42c077a69ad0"
            "be1c92f50e4c9e4b3509c53d000ceb2ea272acaa2984ad50ca2c95720db930254c63a671282895ab"
            "27585edd6ace77"
        ),
        (
            "05f60cfaf714f4220b0428f506f6f1d222d8f91043f70e36ba09f0d1d0da26e835091123fe17f723"
            "0bc0e81bf506ef0a3cf4f621f7fe00c9370ce5d5e321f936330628220840160efce21c1f5f1216e8"
            "2e00e60bd1e9e10433edd23120f51911da2a06c8fbbf22f216f119a51311e635eee90fcad1f5f3f9"
            "e63915e311201ad5372ee538cbe506100110e4ef33a90eeaf8cb34f641c022ff234126fddbf4e723"
            "0eed3ff71b12e4402efa11531b18f2e5ebd1ecff01b11107bfcb2147131a13d8deeafcfe0c29ff30"
            "22d11bfafc42e12c14fad6e50b19adfd17ce131c7d1f15b9"
        ),
    ),
    ResearchVector(
        256,
        1400,
        100,
        (
            "90f47a5fe36de431053ff0fd78786e4a5bbd2960b452f06d6ebd220f1f5191d8a1bedcc80d630edb"
            "9ea5b37c6e35ffc63312b9ad457483e57ec0382a0dd91fc71082ca6d1811a990ad314629320cd936"
            "7d6188c9079cab136f38657f38ce4216b8d86c536c317fd50a63036be33c9ef40ca2c0eb9382269d"
            "050ac477dae4b1e652a683199380a4656eac65f2e0bcb09d46aa8011f5e257beb338c40ca1df0607"
            "e834c12c8f83e3c51966fb5867ab2290f4"
        ),
        (
            "b9c00323ee1dcf30fcc7c3e14b12212a57fd52d0fefe0201f800dda82fc73bead6f0ceb6c9cdf6dc"
            "1023292806fb1b2024241a11ad59b6ea0110c5f9b92b0223eef3d5efe71d14bc513199ef3216182f"
            "facbf101d12428434bd0a7bef555f407f0ee4a1fd7d0ed26328ccde8e9dab2eb122058e6e432ce28"
            "18d924356cfbf8fbf92a41021dfa57e243e7dd072be110f8690e0e97431d08d15a161c3ddc23255f"
            "f5d619b5d2c30c2be9ded7c4dfdbff19de004834bacdc0d728d7c31a4611eb81aadfdbbff4e2e5cb"
            "6cb91236d7f0e3db51001a38dff35000edb025deb610f0018ff89a23c017e5e1d44def192be0fd0f"
            "ed262d0fdf02dafda331151029035e00"
        ),
    ),
    ResearchVector(
        256,
        1536,
        99,
        (
            "b6eb1afcfecadd4f8a41215abec5d712bc1d8d870389df91b62b2f09ce0b84650ac0989ed892a7bc"
            "062ddcff41ae18463bfe7dbc9d4eb4b51da1ceed01170b4d8df105730771c0360f7e96677a7bc6fb"
            "79503ee22e5584c2f4cbdd795ff6e751a4c867c4008e35ebc3c9a7d6ea453025e968ecca52ed24d0"
            "d5c5d4dac86315d23d81ee7fb66d961205a2495608a5fdcde376f6892600621bd5c809492ed010ba"
            "eff84f9cb343f1f1e1f7618747dadcf966edc1c959be67eafbddd7092e59a72cb6eb"
        ),
        (
            "c03128f716c701e6d3f3b8c4d7bf4b3fc9271ada04b2ffed1134fef4e3f3dacced1cfa1505dd03c2"
            "1030b3a91bc1343536edfb2a3009f8b6aaf720e6221454080aea301b2af938020ea1e1082024f307"
            "f3d73614cbecff2bb9efd5cb36c017e24c0df8d50a3e42f1e8f6fa3439169c30fbcb4e4308e34005"
            "09d5eaf929ebd329ed0ebd2d0cfc0f1c1743c202fe09e53c5b1f201a01c30238d5f9b90116c614ba"
            "effe23b8e7eb2706f62fdc0d0decf4db0186c0d8a02729f9e30fe1ce13f3d9fcf5f43431a7bc0c40"
            "fdef14185bfb22ecd106e3f8b710c420c7fcdec503d4f51610f3fffeff38f9ead9d0cc012dfe3312"
            "da1c07e8cafddcd3f00ce3360effc719"
        ),
    ),
)
